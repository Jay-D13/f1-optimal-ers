from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from jsonargparse import ArgumentParser, ActionConfigFile


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
    use_tumftm: bool = False
    plot: bool = True
    save_animation: bool = False
    solver: Literal["nlp"] = "nlp"
    collocation: Literal["euler", "trapezoidal", "hermite_simpson"] = "euler"
    nlp_solver: Literal["auto", "ipopt", "fatrop", "sqpmethod"] = "auto"
    ipopt_linear_solver: str = "mumps"
    ipopt_hessian: Literal["limited-memory", "exact"] = "limited-memory"
    regulations: Literal["2025", "2026"] = "2025"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='F1 ERS Optimal Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        default_env=True,
        env_prefix="F1_",
        epilog="""
        Examples:
        python main.py --track Monaco --plot
        python main.py --track Monza --initial-soc 0.6 --plot --save-animation
        python main.py --track Spa --year 2023 --driver VER --plot
        python main.py --track Monza --laps 10 --final-soc-min 0.45

        # Compare collocation methods:
        python main.py --track Monaco --collocation euler --plot
        python main.py --track Monaco --collocation trapezoidal --plot
        python main.py --track Monaco --collocation hermite_simpson --plot

        # Use a JSON/YAML config file
        python main.py --config configs/example.yaml

        Collocation Methods:
        euler            - 1st order explicit Euler (fastest, least accurate)
        trapezoidal      - 2nd order implicit trapezoidal (good balance)
        hermite_simpson  - 4th order Hermite-Simpson (most accurate, slower)
        """
    )

    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help="Path to JSON/YAML config file with default args",
    )
    parser.add_class_arguments(AppConfig, fail_untyped=False)

    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-save-animation",
        action="store_false",
        dest="save_animation",
        help=argparse.SUPPRESS,
    )
    return parser
