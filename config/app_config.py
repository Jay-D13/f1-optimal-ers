from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
import yaml


@dataclass
class AppConfig:
    track: str = "Monaco"
    year: int = 2024
    driver: str | None = None
    initial_soc: float = 0.5
    final_soc_min: float = 0.3
    ds: float = 5.0
    laps: int = 1
    per_lap_final_soc_min: float | None = None
    enable_tire_degradation: bool = False
    tire_wear_rate_per_lap: float = 0.012
    tire_min_grip_scale: float = 0.88
    flying_lap: bool = True
    use_tumftm: bool = False
    plot: bool = True
    save_animation: bool = False
    solver: Literal["nlp"] = "nlp"
    collocation: Literal["euler", "trapezoidal", "hermite_simpson"] = "euler"
    nlp_solver: Literal["auto", "ipopt", "fatrop", "sqpmethod"] = "auto"
    ipopt_linear_solver: str = "mumps"
    ipopt_hessian: Literal["limited-memory", "exact"] = "limited-memory"
    regulations: Literal["2025", "2026"] = "2025"


def _load_yaml_defaults(path: Path) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


app = typer.Typer(add_completion=False)


@app.command(help="F1 ERS Optimal Control")
def cli(
    # ── Track & data ──────────────────────────────────────────────
    track: Annotated[str, typer.Option(help="Grand Prix name (e.g. Monaco, Monza, Spa)")] = "Monaco",
    year: Annotated[int, typer.Option(help="Season year for FastF1 telemetry")] = 2024,
    driver: Annotated[Optional[str], typer.Option(help="Driver code for telemetry (e.g. VER, LEC)")] = None,
    use_tumftm: Annotated[bool, typer.Option("--use-tumftm/--no-use-tumftm", help="Prefer TUMFTM raceline over FastF1")] = False,

    # ── Energy / SOC ──────────────────────────────────────────────
    initial_soc: Annotated[float, typer.Option(help="Battery state-of-charge at start [0-1]")] = 0.5,
    final_soc_min: Annotated[float, typer.Option(help="Minimum SOC at end of horizon [0-1]")] = 0.3,
    per_lap_final_soc_min: Annotated[Optional[float], typer.Option(help="Optional SOC floor at each lap boundary")] = None,

    # ── Simulation ────────────────────────────────────────────────
    ds: Annotated[float, typer.Option(help="Spatial discretisation step [m]")] = 5.0,
    laps: Annotated[int, typer.Option(help="Number of laps in the NLP horizon")] = 1,
    flying_lap: Annotated[bool, typer.Option("--flying-lap/--no-flying-lap", help="Continuous lap (no standing start)")] = True,
    regulations: Annotated[str, typer.Option(help="Regulation set: 2025 or 2026")] = "2025",

    # ── Tire degradation ──────────────────────────────────────────
    enable_tire_degradation: Annotated[bool, typer.Option("--enable-tire-degradation/--no-tire-degradation", help="Enable grip loss over laps")] = False,
    tire_wear_rate_per_lap: Annotated[float, typer.Option(help="Grip fraction lost per lap")] = 0.012,
    tire_min_grip_scale: Annotated[float, typer.Option(help="Minimum grip scale floor (0, 1]")] = 0.88,

    # ── Solver ────────────────────────────────────────────────────
    solver: Annotated[str, typer.Option(help="Solver type (nlp)")] = "nlp",
    collocation: Annotated[str, typer.Option(help="Integration: euler, trapezoidal, hermite_simpson")] = "euler",
    nlp_solver: Annotated[str, typer.Option(help="NLP backend: auto, ipopt, fatrop, sqpmethod")] = "auto",
    ipopt_linear_solver: Annotated[str, typer.Option(help="Ipopt linear solver (e.g. mumps, ma97)")] = "mumps",
    ipopt_hessian: Annotated[str, typer.Option(help="Ipopt Hessian: limited-memory or exact")] = "limited-memory",

    # ── Output ────────────────────────────────────────────────────
    plot: Annotated[bool, typer.Option("--plot/--no-plot", help="Generate visualisation plots")] = True,
    save_animation: Annotated[bool, typer.Option("--save-animation/--no-save-animation", help="Save animated lap GIF")] = False,

    # ── Config file ───────────────────────────────────────────────
    config: Annotated[Optional[Path], typer.Option(help="Path to YAML config file")] = None,
):
    """Entry point — build AppConfig from CLI args (with optional YAML defaults) and run."""
    from main import main as run_main

    # Collect all explicitly-provided CLI values
    # typer passes through defaults for unset options, so we build the
    # full dict then overlay YAML defaults underneath.
    cli_values = {
        "track": track, "year": year, "driver": driver, "use_tumftm": use_tumftm,
        "initial_soc": initial_soc, "final_soc_min": final_soc_min,
        "per_lap_final_soc_min": per_lap_final_soc_min, "ds": ds, "laps": laps,
        "flying_lap": flying_lap, "regulations": regulations,
        "enable_tire_degradation": enable_tire_degradation,
        "tire_wear_rate_per_lap": tire_wear_rate_per_lap,
        "tire_min_grip_scale": tire_min_grip_scale, "solver": solver,
        "collocation": collocation, "nlp_solver": nlp_solver,
        "ipopt_linear_solver": ipopt_linear_solver, "ipopt_hessian": ipopt_hessian,
        "plot": plot, "save_animation": save_animation,
    }

    if config is not None:
        yaml_defaults = _load_yaml_defaults(config)
        # YAML provides defaults; CLI args override
        merged = {**yaml_defaults, **cli_values}
    else:
        merged = cli_values

    args = AppConfig(**merged)
    run_main(args)
