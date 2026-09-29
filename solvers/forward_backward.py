"""
Forward-backward speed profile on the shared car model (models/car.py).

The NLP has the grip limits inside, so this profile is only its initial guess and a quick reference
(e.g. the "no ERS" lap). Based on TUMFTM's laptime simulation approach.
"""
import numpy as np

from models.car import V_MAX, deploy_power_limit
from solvers import VelocityProfileSolver, VelocityProfile

V_MIN = 5.0                              # Lowest speed the passes produce (m/s)
V_GRID = np.linspace(V_MIN, V_MAX, 64)   # Speeds at which the acceleration limits are tabulated
_BISECTIONS = 40


class ForwardBackwardSolver(VelocityProfileSolver):
    """
    Grip- and power-limited speed profile.

    Every point's speed is capped by the highest speed the tyres allow there, then a forward pass
    accelerates as hard as grip and power allow and a backward pass brakes as hard as grip and the
    brakes allow. Both passes use the car model's per-axle friction ellipses and load transfer.
    With use_ers_power, the MGU-K adds its deploy limit on top of the ICE, with no energy limit.
    """

    def __init__(self, vehicle_model, track_model, use_ers_power: bool = False):
        super().__init__(vehicle_model, track_model)
        self.use_ers_power = use_ers_power

    def solve(self, flying_lap: bool = False) -> VelocityProfile:
        """
        Solve for the speed profile.

        Args:
            flying_lap: If True, simulates 2 laps and returns the second, so V_start == V_end
        """
        if flying_lap:
            print("==== Solving grip-limited velocity profile (Flying Lap)...====")
            return self._solve_flying()

        print("==== Solving grip-limited velocity profile (Standing Start)...====")
        td = self.track.track_data
        return self._solve_core(td.s, td.radius, td.gradient, td.ds)

    def _solve_flying(self) -> VelocityProfile:
        """Solve two consecutive laps; the first is the run-up for the second."""
        td = self.track.track_data
        N = len(td.s)
        s_double = np.concatenate([td.s, td.s + td.total_length])
        profile = self._solve_core(s_double, np.tile(td.radius, 2), np.tile(td.gradient, 2), td.ds)

        v_flying = profile.v[N:]
        t_flying = np.concatenate([[0.0], np.cumsum(td.ds / np.maximum(0.5 * (v_flying[1:] + v_flying[:-1]), 1.0))])
        print(f"   ✓ Flying lap time: {t_flying[-1]:.3f}s")
        print(f"   Entry Speed: {v_flying[0] * 3.6:.1f} km/h")

        return VelocityProfile(s=td.s, v=v_flying, a_x=profile.a_x[N:], t=t_flying, lap_time=t_flying[-1])

    def _solve_core(self, s, radius, gradient, ds) -> VelocityProfile:
        """Forward-backward integration over the given points."""
        car = self.vehicle.car
        N = len(s)
        kappa = 1.0 / np.abs(radius)
        gradient = np.asarray(gradient, dtype=float)
        w = car.aero_mode(radius)

        v_apex = self._cornering_speeds(kappa, gradient, w)
        a_max, a_min = self._acceleration_limits(kappa, gradient, w)
        print(f"   Apex speeds: {v_apex.min():.1f} - {v_apex.max():.1f} m/s")

        # Forward pass: accelerate as hard as possible
        v_fwd = np.empty(N)
        v_fwd[0] = v_apex[0]
        for i in range(N - 1):
            a = np.interp(v_fwd[i], V_GRID, a_max[i])
            v_fwd[i + 1] = min(np.sqrt(max(v_fwd[i] ** 2 + 2.0 * a * ds, V_MIN**2)), v_apex[i + 1])

        # Backward pass: brake as late as possible
        v_bwd = np.empty(N)
        v_bwd[-1] = v_apex[-1]
        for i in range(N - 1, 0, -1):
            a = np.interp(v_bwd[i], V_GRID, a_min[i])
            v_bwd[i - 1] = min(np.sqrt(max(v_bwd[i] ** 2 - 2.0 * a * ds, V_MIN**2)), v_apex[i - 1])

        v_final = np.minimum(v_fwd, v_bwd)
        a_x = np.append((v_final[1:] ** 2 - v_final[:-1] ** 2) / (2.0 * ds), 0.0)
        t = np.concatenate([[0.0], np.cumsum(ds / np.maximum(0.5 * (v_final[1:] + v_final[:-1]), 1.0))])

        print(f"   ✓ Theoretical lap time: {t[-1]:.3f}s")
        print(f"   Velocity range: {v_final.min():.1f} - {v_final.max():.1f} m/s")
        print(f"   Acceleration range: {a_x.min():.1f} - {a_x.max():.1f} m/s²")

        return VelocityProfile(s=s, v=v_final, a_x=a_x, t=t, lap_time=t[-1])

    def _max_power(self, v):
        """Propulsive power available at speed v (W): ICE, plus the MGU-K deploy limit with use_ers_power."""
        veh = self.vehicle.vehicle
        if not self.use_ers_power:
            return np.full_like(np.asarray(v, dtype=float), veh.pow_max_ice)
        deploy = np.array([float(deploy_power_limit(x, self.vehicle.ers)) for x in np.ravel(v)]).reshape(np.shape(v))
        return veh.pow_max_ice + deploy

    def _cornering_speeds(self, kappa, gradient, w, hold: bool = False):
        """
        Highest speed at each point that the tyres allow.

        By default the car only has to pass the point while coasting, so the tyres carry no longitudinal
        force. That is what limits it at a curvature peak; in a long corner the forward pass then slows it
        to the speed it can hold. With hold=True the car must hold its speed there (a_x = 0, the rear tyres
        also pushing against drag): the steady cornering speed.
        """
        car = self.vehicle.car

        def ok(v):
            a_x = 0.0 if hold else -car.resistance(v, kappa, gradient, w) / car.mass
            return car.is_feasible(v, kappa, gradient, w, a_x, np.inf)

        lo = np.full(len(kappa), V_MIN)
        hi = np.full(len(kappa), V_MAX)
        flat_out = ok(hi)
        for _ in range(_BISECTIONS):
            mid = 0.5 * (lo + hi)
            mid_ok = ok(mid)
            lo = np.where(mid_ok, mid, lo)
            hi = np.where(mid_ok, hi, mid)
        return np.where(flat_out, V_MAX, lo)

    def _acceleration_limits(self, kappa, gradient, w):
        """
        Highest and lowest feasible acceleration at each point, tabulated over V_GRID: arrays of shape (N, len(V_GRID)).

        Where no acceleration is feasible (the axles can't carry the lateral force at that speed), both limits
        are the coasting deceleration; the cornering-speed cap keeps the passes away from those speeds.
        """
        car = self.vehicle.car
        v = V_GRID[None, :]
        kappa, gradient, w = kappa[:, None], gradient[:, None], w[:, None]
        p_max = self._max_power(V_GRID)[None, :]

        def ok(a):
            return car.is_feasible(v, kappa, gradient, w, a, p_max)

        resistance = car.resistance(v, kappa, gradient, w)
        a_power = (p_max / v - resistance) / car.mass                                 # Full power, if grip allows
        a_brakes = -(car.vehicle.max_brake_force + resistance) / car.mass              # Full brakes, if grip allows
        a_coast = np.broadcast_to(-resistance / car.mass, a_power.shape)

        # A feasible acceleration to bisect from: holding speed, or else coasting
        hold_ok = ok(np.zeros_like(a_power)) & (a_power >= 0.0)
        coast_ok = ok(a_coast)
        inside = np.where(hold_ok, 0.0, a_coast)
        any_ok = hold_ok | coast_ok

        def bisect(feasible_end, infeasible_end):
            good, bad = feasible_end.copy(), infeasible_end.copy()
            for _ in range(_BISECTIONS):
                mid = 0.5 * (good + bad)
                mid_ok = ok(mid)
                good = np.where(mid_ok, mid, good)
                bad = np.where(mid_ok, bad, mid)
            return good

        a_max = np.where(ok(a_power), a_power, bisect(inside, a_power))
        a_min = np.where(ok(a_brakes), a_brakes, bisect(inside, a_brakes))
        return np.where(any_ok, a_max, a_coast), np.where(any_ok, a_min, a_coast)
