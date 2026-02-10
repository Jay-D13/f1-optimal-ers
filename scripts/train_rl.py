"""
Minimal ERS RL training entrypoint.

This script trains a CleanRL-style PPO policy in a single-lap environment where:
- ERS command is learned by RL
- throttle/brake are provided by a reference tracking controller
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import VehicleConfig, get_ers_config, get_vehicle_config
from models import F1TrackModel, VehicleDynamicsModel
from rl.env import ERSLapEnv
from rl.policy import PPOPolicy, load_policy_checkpoint, save_policy_checkpoint
from simulation import LapSimulator
from solvers import ForwardBackwardSolver
from strategies.baselines import TrackingStrategy
from strategies.rl_strategy import RLERSStrategy


def build_vehicle_config(track_name: str, regulations: str) -> VehicleConfig:
    track_configs = {
        "monaco": VehicleConfig.for_monaco,
        "monza": VehicleConfig.for_monza,
        "montreal": VehicleConfig.for_montreal,
        "spa": VehicleConfig.for_spa,
        "silverstone": VehicleConfig.for_silverstone,
    }
    base_cfg = track_configs.get(track_name.lower(), VehicleConfig)()
    return get_vehicle_config(regulations, base=base_cfg)


def load_track(args: argparse.Namespace) -> tuple[F1TrackModel, str]:
    track = F1TrackModel(year=args.year, gp=args.track, ds=args.ds)

    if args.raceline_csv is not None:
        raceline_path = Path(args.raceline_csv)
        if not raceline_path.exists():
            raise FileNotFoundError(f"raceline file not found: {raceline_path}")
        track.load_from_tumftm_raceline(str(raceline_path))
        return track, "TUMFTM"

    _, driver = track.load_from_fastf1(driver=args.driver)
    return track, driver


def print_result(label: str, result) -> None:
    print(f"\n{label}")
    print(f"  Lap complete:      {result.completed}")
    print(f"  Lap time:          {result.lap_time:.3f}s")
    print(f"  Final SOC:         {result.final_soc * 100.0:.1f}%")
    print(f"  Energy deployed:   {result.energy_deployed / 1e6:.3f} MJ")
    print(f"  Energy recovered:  {result.energy_recovered / 1e6:.3f} MJ")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_env(env_kwargs: dict, seed: int):
    def thunk():
        import gymnasium as gym

        env = ERSLapEnv(**env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def train(args: argparse.Namespace) -> Path:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError(
            "RL dependencies missing. Install with:\n"
            "  uv add gymnasium torch\n"
            "or\n"
            "  pip install gymnasium torch"
        ) from exc

    _set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 70)
    print("RL TRAINING (CLEANRL-STYLE PPO ERS)")
    print("=" * 70)
    print(f"Device: {device}")

    ers_cfg = get_ers_config(args.regulations)
    veh_cfg = build_vehicle_config(args.track, args.regulations)
    track, driver = load_track(args)
    print(f"Track source driver: {driver}")

    vehicle_model = VehicleDynamicsModel(veh_cfg, ers_cfg)
    fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
    reference_profile = fb_solver.solve(flying_lap=True)

    env_kwargs = dict(
        vehicle_model=vehicle_model,
        track_model=track,
        reference_profile=reference_profile,
        dt=args.dt,
        initial_soc=args.initial_soc,
        initial_velocity=args.initial_velocity,
        final_soc_target=args.final_soc_target,
        max_time=args.max_time,
    )

    envs = gym.vector.SyncVectorEnv(
        [_make_env(env_kwargs, args.seed + i) for i in range(args.num_envs)]
    )

    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise RuntimeError("Only continuous action spaces are supported")

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    policy = PPOPolicy(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.timesteps // batch_size
    if num_updates < 1:
        raise ValueError("timesteps too small. Increase timesteps or decrease batch size.")

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    start_time = time.time()
    print(
        f"Training for {args.timesteps:,} steps "
        f"({num_updates} updates, batch={batch_size}, minibatch={minibatch_size})"
    )

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_obs)
                action = torch.clamp(action, -1.0, 1.0)
                values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, term_np, trunc_np, _ = envs.step(action.cpu().numpy())
            done_np = np.logical_or(term_np, trunc_np)

            rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done_np.astype(np.float32), dtype=torch.float32, device=device)

            episode_returns += reward_np
            episode_lengths += 1
            for env_i in np.where(done_np)[0]:
                completed_returns.append(float(episode_returns[env_i]))
                completed_lengths.append(int(episode_lengths[env_i]))
                episode_returns[env_i] = 0.0
                episode_lengths[env_i] = 0

        with torch.no_grad():
            next_value = policy.get_value(next_obs).squeeze(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_non_terminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_actions = actions.reshape((-1, action_dim))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clip_fracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                approx_kl = ((ratio - 1.0) - logratio).mean().item()
                if approx_kl > args.target_kl:
                    break

        if update % args.log_every == 0 or update == 1 or update == num_updates:
            avg_return = float(np.mean(completed_returns[-20:])) if completed_returns else float("nan")
            avg_len = float(np.mean(completed_lengths[-20:])) if completed_lengths else float("nan")
            sps = int((update * batch_size) / max(time.time() - start_time, 1e-6))
            print(
                f"[{update:04d}/{num_updates}] "
                f"avg_ep_return={avg_return:.3f} avg_ep_len={avg_len:.1f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} clip_frac={np.mean(clip_fracs):.3f} sps={sps}"
            )

    out_path = Path(args.output_model)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".pt")
    save_policy_checkpoint(policy, out_path)
    print(f"Saved model: {out_path}")

    print("\nRunning quick rollout comparison...")
    loaded_policy = load_policy_checkpoint(out_path, device="cpu")

    sim_rl = LapSimulator(vehicle_model, track, controller=RLERSStrategy(
        vehicle_config=veh_cfg,
        ers_config=ers_cfg,
        track_model=track,
        policy=loaded_policy,
        reference_profile=reference_profile,
        dt=args.dt,
        final_soc_target=args.final_soc_target,
    ), dt=args.dt)
    rl_result = sim_rl.simulate_lap(
        initial_soc=args.initial_soc,
        initial_velocity=args.initial_velocity,
        max_time=args.max_time,
    )

    sim_baseline = LapSimulator(
        vehicle_model,
        track,
        controller=TrackingStrategy(veh_cfg, ers_cfg, track, reference_profile=reference_profile),
        dt=args.dt,
    )
    baseline_result = sim_baseline.simulate_lap(
        initial_soc=args.initial_soc,
        initial_velocity=args.initial_velocity,
        max_time=args.max_time,
    )

    envs.close()
    print_result("RL Policy", rl_result)
    print_result("Baseline Tracking (No ERS)", baseline_result)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal CleanRL-style PPO ERS policy.")

    parser.add_argument("--track", type=str, default="Monaco")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--driver", type=str, default=None)
    parser.add_argument("--regulations", type=str, default="2026", choices=["2025", "2026"])
    parser.add_argument("--raceline-csv", type=str, default=None, help="Path to TUMFTM raceline CSV")

    parser.add_argument("--ds", type=float, default=5.0, help="Track discretization step (m)")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation integration step (s)")
    parser.add_argument("--max-time", type=float, default=220.0)

    parser.add_argument("--initial-soc", type=float, default=0.5)
    parser.add_argument("--initial-velocity", type=float, default=30.0)
    parser.add_argument("--final-soc-target", type=float, default=0.3)

    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=5)

    parser.add_argument("--output-model", type=str, default="results/models/cleanrl_ppo_ers.pt")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
