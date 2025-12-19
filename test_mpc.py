"""
Quick test to verify MPC multi-lap works
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from models import F1TrackModel, VehicleDynamicsModel
from config import ERSConfig, VehicleConfig
from solvers import ForwardBackwardSolver, SpatialNLPSolver

# Import the FIXED versions
from controllers import ERSModelPredictiveController
from simulation import MultiLapSimulator, OpenLoopController

print("Testing MPC Multi-Lap System (FIXED VERSION)...")
print("="*60)

# 1. Create simple track
print("\n1. Creating track model...")
track = F1TrackModel(year=2024, gp='Monaco', ds=10.0)  # Coarse for speed

# Use TUMFTM if available, otherwise will need FastF1
tumftm_path = Path('data/racelines/montreal.csv')
if tumftm_path.exists():
    track.load_from_tumftm_raceline(str(tumftm_path))
    print("   ✓ Track loaded")
else:
    print("   ⚠ No TUMFTM raceline, test will use FastF1 (slower)")
    print("   Please add montreal.csv to data/racelines/ for faster testing")
    sys.exit(0)

# 2. Create models
print("\n2. Creating vehicle model...")
vehicle_config = VehicleConfig()
ers_config = ERSConfig()
vehicle_model = VehicleDynamicsModel(vehicle_config, ers_config)
print("   ✓ Vehicle model created")

# IMPORTANT: Compute speed limits for the track!
print("   Computing speed limits...")
v_max = track.compute_speed_limits(vehicle_config)
print(f"   ✓ Speed limits computed: {v_max.min():.1f} - {v_max.max():.1f} m/s")

# 3. Compute reference (quick version)
print("\n3. Computing reference trajectory...")
fb_solver = ForwardBackwardSolver(vehicle_model, track, use_ers_power=True)
velocity_profile = fb_solver.solve()

nlp_solver = SpatialNLPSolver(vehicle_model, track, ers_config, ds=10.0)
reference_trajectory = nlp_solver.solve(
    v_limit_profile=velocity_profile.v,
    initial_soc=0.5,
    final_soc_min=0.4,
    energy_limit=ers_config.deployment_limit_per_lap,
)
print(f"   ✓ Reference lap time: {reference_trajectory.lap_time:.3f}s")
print(f"   Reference throttle: min={reference_trajectory.throttle_opt.min():.2f}, "
      f"max={reference_trajectory.throttle_opt.max():.2f}, "
      f"mean={reference_trajectory.throttle_opt.mean():.2f}")

# 4. Test MPC controller
print("\n4. Testing MPC controller...")
mpc = ERSModelPredictiveController(
    vehicle_model=vehicle_model,
    track_model=track,
    ers_config=ers_config,
    horizon_segments=30,  # Small for speed
    verbose=False
)
mpc.set_reference(reference_trajectory)

# Test one MPC step
control, info = mpc.solve_step(
    position=100.0,
    velocity=40.0,
    soc=0.5,
    tire_degradation=0.95,  # 5% degraded
    fuel_mass=798.0
)
print(f"   ✓ MPC step solved in {info['solve_time']:.3f}s")
print(f"     Status: {info['status']}")
print(f"     Control: P_ers={control[0]/1000:.1f}kW, throttle={control[1]:.2f}, brake={control[2]:.2f}")

# CHECK: Is throttle reasonable?
if control[1] < 0.2:
    print(f"   ⚠ WARNING: Throttle is very low ({control[1]:.2f})")
    print("   This might cause slow lap times!")
else:
    print(f"   ✓ Throttle looks reasonable")

# 5. Test multi-lap simulator (just 2 laps for speed)
print("\n5. Testing multi-lap simulator...")

# Use MPC call interval to reduce number of MPC solves
simulator = MultiLapSimulator(
    vehicle_model, 
    track, 
    dt=0.1,  # 100ms timestep
    mpc_call_interval=50.0  # Call MPC every 50m instead of every timestep
)

print("\n   Testing Open-Loop (2 laps)...")
open_loop = OpenLoopController(reference_trajectory)
result_ol = simulator.simulate_race(
    controller=open_loop,
    n_laps=2,
    initial_soc=0.5,
    tire_deg_rate=0.03,
    strategy_name="Open-Loop Test"
)
print(f"   ✓ Open-loop: {result_ol.total_time:.3f}s")

print("\n   Testing MPC (2 laps)...")
result_mpc = simulator.simulate_race(
    controller=mpc,
    n_laps=2,
    initial_soc=0.5,
    tire_deg_rate=0.03,
    reference_trajectory=reference_trajectory,
    strategy_name="MPC Test"
)
print(f"   ✓ MPC: {result_mpc.total_time:.3f}s")

# 6. Summary
print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)
result_ol.print_summary()
result_mpc.print_summary()

print(f"Open-Loop:  {result_ol.total_time:.3f}s")
print(f"MPC:        {result_mpc.total_time:.3f}s")
diff = result_ol.total_time - result_mpc.total_time
print(f"Difference: {diff:+.3f}s")

# Check if reasonable
if result_mpc.total_time < 200:  # Less than 200s for 2 laps ~= 100s/lap is reasonable
    print(f"\n✓ MPC lap times look REASONABLE!")
    if abs(diff) < 10.0:
        print("✓ MPC and Open-Loop are comparable - TEST PASSED!")
    else:
        print(f"⚠ Some gap between MPC and Open-Loop ({diff:.1f}s)")
else:
    print(f"\n✗ MPC lap times still too slow ({result_mpc.total_time:.1f}s)")
    print("  Expected: ~150-180s for 2 laps on Montreal")
    print("  Debugging info:")
    for lap in result_mpc.lap_results:
        print(f"    Lap {lap.lap_number}: {lap.lap_time:.1f}s, avg_v={lap.avg_velocity:.1f}m/s")

print("="*60)