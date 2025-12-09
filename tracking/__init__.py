"""
Online Tracking Controllers

These controllers track an offline-computed reference trajectory in real-time.

Available trackers:
- MPCTracker: Nonlinear MPC for reference tracking
- MPCTrackerRobust: MPC with soft constraints
- LTVMPCTracker: Linearized MPC for faster solves

Usage:
    1. Compute optimal trajectory offline using SpatialNLPSolver
    2. Create MPCTracker and set reference
    3. In simulation loop, call solve_mpc_step() at each timestep
"""

from .mpc_tracker import (
    MPCTracker,
    MPCTrackerRobust,
    LTVMPCTracker,
)

__all__ = [
    'MPCTracker',
    'MPCTrackerRobust',
    'LTVMPCTracker',
]
