<h1 align="center">🏎️ F1 ERS Optimal Control</h1>

<p align="center">
  <strong>Optimal Energy Recovery System Usage in Formula 1 Hybrid Powertrains for Lap Time Minimization</strong>
</p>

<p align="center">
  <em>A physics-based optimization framework for computing lap-time-optimal ERS deployment strategies using a spatial-domain nonlinear program with per-axle tyre grip.</em>
</p>

---

<p align="center">
  <img src="docs/poster.png" alt="F1 ERS Optimal Control - Project Poster" width="50%"/>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Motivation & Goals](#motivation--goals)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [The Experiment](#the-experiment)
- [Command Line Arguments](#command-line-arguments)
- [Data Sources](#data-sources)
- [Baseline Strategies](#baseline-strategies)
- [Generating Poster Plots](#generating-poster-plots)
- [Results & Outputs](#results--outputs)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

This project computes lap-time-optimal Formula 1 Energy Recovery System (ERS) strategies on a fixed racing line:

1. **Car model** (`models/car.py`): a point mass with per-axle tyre forces (load-sensitive friction ellipses, load transfer, rear-wheel drive), aero drag and downforce, ICE and MGU-K power, and the battery. Every solver uses these same equations.
2. **Velocity profiling**: a forward-backward pass on the car model gives a quick grip- and power-limited speed profile, used as the optimizer's initial guess and as a no-ERS reference
3. **ERS optimization**: a spatial NLP (Nonlinear Programming) solver finds the speed, throttle, brake and battery deployment/recovery that minimize lap time, with the tyre grip limits inside the optimization

The framework supports both **2025 regulations** (120kW MGU-K, 4MJ deployment limit) and **upcoming 2026 regulations** (350kW MGU-K, 8.5MJ recovery, no MGU-H).

---

## Motivation & Goals

### Inspiration

Modern F1 cars are hybrid powertrains producing ~1000HP combined from:
- **Internal Combustion Engine (ICE)**: ~575kW (770HP)
- **MGU-K (Kinetic)**: 120kW electrical motor/generator
- **MGU-H (Heat)**: Unlimited power turbo recovery (removed in 2026)

The ERS system can deploy up to **4MJ per lap** but is limited in how much it can harvest (2MJ from braking). This creates a complex optimization problem: *when* and *where* should the driver deploy electrical power to minimize lap time?

### Goals Achieved

- **Validated Physics Models** - Implementation aligns with TUMFTM and Oxford academic approaches  
- **Forward-Backward Solver** - Grip- and power-limited speed profiles on the same car model, for the NLP's initial guess  
- **Spatial NLP Optimization** - Direct collocation with CasADi/IPOPT for globally optimal ERS deployment (higher order of Gauss-Legendre upgrade in the works)
- **New Regulation Support** - Compare 2025 vs 2026 powertrain regulations  
- **Real Telemetry Integration** - Loading actual F1 data via FastF1 API  
- **Baseline Strategy Comparison** - Evaluate rule-based strategies against optimal  
---

## Installation

### Prerequisites

- **Python** 3.10–3.12 (recommended: 3.11)
- **uv** (package manager by Astral) or pip
- Internet access (FastF1 downloads telemetry data)
- Optional: HSL MA97 linear solver for faster IPOPT performance

### Install with uv (Recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/f1-ers-optimal-control.git
cd f1-ers-optimal-control

# Create virtual environment and install dependencies
uv sync

# Set up FastF1 cache directory
mkdir -p data/cache
```

### Install with pip

```bash
# Clone and enter directory
git clone https://github.com/yourusername/f1-ers-optimal-control.git
cd f1-ers-optimal-control

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up cache
mkdir -p data/cache
```

---

## Quick Start

### 1. Unified Startup (Recommended)

Run the full stack (Backend + Frontend) with a single command:

```bash
./start.sh
```

This will launch:
-   **Frontend**: http://localhost:5173
-   **Backend**: http://localhost:8000

**Optional Arguments:**
You can configure the ports if needed:
```bash
./start.sh --backend-port 8001 --frontend-port 3000
```

### 2. Manual CLI Usage

If you only want to run the optimization core (without the web UI), you can use the CLI directly:

#### Run the optimizer on Monaco (default)

```bash
uv run python main.py --track Monaco --plot
```

#### Run with animation output

```bash
uv run python main.py --track Monaco --plot --save-animation
```

NOTE: Saving animations is slow due to GIF encoding.

#### Compare 2025 vs 2026 regulations

```bash
# 2025 regulations (120kW ERS)
uv run python main.py --track Monza --regulations 2025 --plot

# 2026 regulations (350kW ERS)
uv run python main.py --track Monza --regulations 2026 --plot
```

#### Run multi-lap NLP race strategy

```bash
# 10-lap horizon with final and per-lap SOC constraints
uv run python main.py --track Monza --laps 10 --final-soc-min 0.45 --per-lap-final-soc-min 0.40
```

### Run the tests

```bash
uv run python -m unittest discover -s tests

# Also solve every bundled track under both rule sets (~2 min)
RUN_TRACK_SWEEP=1 uv run python -m unittest discover -s tests
```

### Apple Silicon Note (M5 tested)

- Ipopt runs on Apple Silicon. The solver sets MUMPS's AMD ordering (`mumps_pivot_order = 0`) on every platform, because the default ordering in CasADi's bundled MUMPS segfaults on Apple Silicon.
- `--nlp-solver auto` (the default) means Ipopt everywhere. `fatrop` is available as an opt-in, but it barely progresses on this problem.

On first run, FastF1 will download session telemetry data to `data/cache/`. This may take a few minutes.

---

## The Experiment

### Problem Formulation

The optimization minimizes **lap time** in the spatial domain:

```
minimize    T = ∫(1/v)ds                    (lap time integral)

subject to: 
    dv/ds = a_x(v, controls, κ(s)) / v      (car model: power, brakes, drag, rolling resistance)
    (F_x/F_x,max)² + (F_y/F_y,max)² ≤ 1     (friction ellipse on each axle, load-sensitive)
    dSOC/ds = f(P_ers, v, η)                (battery dynamics)
    ∫ P_deploy ds ≤ 4 MJ                    (regulatory deployment limit)
    ∫ P_harvest ds ≤ 2 MJ                   (regulatory recovery limit)
    SOC_min ≤ SOC ≤ SOC_max                 (battery health limits)
    |P_ers| ≤ P_max                         (power limits)
```

### Two-Phase Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               PHASE 1: Initial guess                         │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │ Track Data  │───▶│ Forward-Backward │───▶│ v(s) guess │  │
│  │ (curvature) │    │ (car model)      │    │ (no ERS)   │  │
│  └─────────────┘    └──────────────────┘    └────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               PHASE 2: ERS Optimization                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │ v(s) guess  │───▶│  Spatial NLP     │───▶│ Optimal    │  │
│  │ + car model │    │  (CasADi/IPOPT)  │    │ trajectory │  │
│  │ + ERS cfg   │    │  grip inside     │    │            │  │
│  └─────────────┘    └──────────────────┘    └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Why Spatial Domain?

The spatial formulation (optimizing over distance `s` rather than time `t`) offers several advantages:

1. **Fixed endpoint**: Track length L is known; lap time T is what we're minimizing
2. **Natural constraints**: Curvature κ(s), track width, banking are functions of position
3. **No singularities** (for racing): Minimum speeds ~50 km/h avoid 1/v divergence
4. **Computational efficiency**: Significant speedups reported by TUMFTM

---

## Command Line Arguments

```bash
python main.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | path | `None` | Load defaults from a JSON/YAML config file |
| `--track` | str | `Monaco` | Track name (Monaco, Monza, Spa, Montreal, Silverstone) |
| `--year` | int | `2024` | Season year for FastF1 telemetry data |
| `--driver` | str | `None` | Driver code (VER, HAM, LEC, etc.) - uses fastest if not specified |
| `--initial-soc` | float | `0.5` | Initial battery state of charge (0.0-1.0) |
| `--final-soc-min` | float | `0.3` | Minimum final SOC constraint (0.0-1.0) |
| `--ds` | float | `5.0` | Spatial discretization step in meters |
| `--laps` | int | `1` | Number of consecutive laps in the NLP horizon |
| `--per-lap-final-soc-min` | float | `None` | Optional SOC floor at each lap boundary |
| `--regulations` | str | `2025` | Regulation set: `2025` or `2026` |
| `--enable-tire-degradation/--no-tire-degradation` | flag | `False` | Scalar tire model: lose grip lap by lap |
| `--tire-wear-rate-per-lap` | float | `0.012` | Scalar tire model: grip fraction lost per lap |
| `--tire-min-grip-scale` | float | `0.88` | Scalar tire model: minimum grip scale |
| `--tire-model` | str | `scalar` | `scalar` (per-lap grip scale) or `dynamic` (tire temperature and wear states; multi-lap only) |
| `--tire-compound` | str | `medium` | Dynamic tire model: `soft`, `medium` or `hard` |
| `--ambient-temp-c` | float | `25.0` | Dynamic tire model: air temperature (°C) |
| `--track-temp-c` | float | `35.0` | Dynamic tire model: track temperature (°C) |
| `--tire-init-temp-c` | float | `80.0` | Dynamic tire model: tire temperature at the start (°C) |
| `--collocation` | str | `trapezoidal` | Integration method: `trapezoidal`, `hermite_simpson`, `euler` |
| `--session` | str | `qualifying` | 2026 energy rules: `qualifying` (Overtake curve, per-event recharge cap, 4 MJ window starting full, run-up from the last corner, ramp-down rules) or `race` |
| `--event` | str | from `--track` | 2026 round number or name for the event's energy limits |
| `--nlp-solver` | str | `auto` | NLP backend: `auto` (= `ipopt`), `ipopt`, `fatrop`, or `sqpmethod` |
| `--ipopt-linear-solver` | str | `mumps` | Ipopt linear solver backend (advanced) |
| `--ipopt-hessian` | str | `exact` | Ipopt Hessian mode: `exact`, or `limited-memory` (faster per iteration, but can stop seconds away from the optimum) |
| `--flying-lap/--no-flying-lap` | flag | `True` | Continuous lap (no standing start) |
| `--use-tumftm/--no-use-tumftm` | flag | `False` | Use the TUMFTM raceline, placed on the FastF1 session (height, timing line, Straight Mode zones); flat if the session is unavailable or its layout has changed |
| `--plot/--no-plot` | flag | `True` | Enable or disable visualization plots |
| `--save-animation/--no-save-animation` | flag | `False` | Enable or disable animated lap visualization |
| `--solver` | str | `nlp` | Solver type (nlp is fully implemented) |

### Examples

```bash
# Load settings from a config file
python main.py --config docs/configs/example.yaml

# Monaco with specific driver
python main.py --track Monaco --driver LEC --plot

# Monza with aggressive energy use (start at 60%, end at 40%)
python main.py --track Montreal --initial-soc 0.6 --final-soc-min 0.4 --plot

# Spa with 2026 regulations
python main.py --track Spa --regulations 2026 --plot --save-animation

# Multi-lap stint optimization (race strategy)
python main.py --track Monza --laps 12 --initial-soc 0.55 --final-soc-min 0.45 --per-lap-final-soc-min 0.35

# Explicit backend choice (auto = ipopt)
python main.py --track Monza --ds 5 --laps 10 --nlp-solver ipopt

# Use TUMFTM raceline instead of FastF1
python main.py --track Monaco --use-tumftm --plot

# Compare collocation methods (accuracy vs speed tradeoff)
python main.py --track Monaco --collocation euler --plot           # Fast, 1st order
python main.py --track Monaco --collocation trapezoidal --plot     # Balanced, 2nd order  
python main.py --track Monaco --collocation hermite_simpson --plot # Accurate, 4th order
```

### Preset + Config Precedence

Resolution order is:

1. Built-in defaults
2. Config file (`--config`)
3. Explicit CLI flags (highest priority)

### Config File Format

Use JSON or YAML with keys that match the CLI option names, with underscores (`initial_soc` for `--initial-soc`). Unknown keys are rejected.

```yaml
track: Monza
year: 2024
initial_soc: 0.55
final_soc_min: 0.45
collocation: trapezoidal
```

---

## Data Sources

The project supports two primary data sources, and **the choice significantly affects results**:

### 1. FastF1 Telemetry (Default)

A line fitted to the position samples of every clean lap of a session (all drivers). **Caveat:** the live-timing positions are snapped to the timing provider's own map of the circuit (all laps lie within about 2 cm of each other), so this is that map line, not the line the cars drive. It sits near the centreline at some circuits (Shanghai, Suzuka) and near a racing line at others (Monza, Spa), and has sharp polyline corners in places (Shanghai T16). Its elevation, timing line and corner positions are sound; its curvature is not a driven line.

```python
track.load_from_fastf1(driver='VER')   # driver only picks the lap kept for plots
```

- The event is found by round number or exact name (`--track 13`, `Monza`, `Catalunya`); ambiguous names such as `Spain` in 2026 stop with the list of rounds.
- Clean laps: within 107 % of the fastest, no pit laps, no deleted laps, green track status. Frozen 2026 car-data blocks (throttle ≥ 104 with the brake on) are dropped.
- Each coordinate is a penalised periodic spline of lap distance. The horizontal smoothing follows the speed (0.8 s of travel, at least 20 m), so noise doesn't become curvature on the straights; the elevation uses 80 m. Samples far from the fit are dropped between rounds (the height feed puts some samples on the wrong level where a track crosses itself).
- The result has curvature, gradient and vertical curvature on a 1 m grid, starts at the timing line, and is cached in `data/cache/geometry/`.
- For 2026 events, the FIA Straight Mode zones (`config/events.py`) are placed from FastF1's corner markers; each zone ends at the next corner.
- Two geometries built from disjoint halves of the drivers give lap times within 0.06 % on all 15 rounds of 2026 (expected, given the snapping).

### 2. TUMFTM Racelines

Minimum-curvature optimal racing lines from [TUMFTM's racetrack database](https://github.com/TUMFTM/racetrack-database):

```python
track.load_from_tumftm_raceline('data/racelines/monza.csv')                    # flat, in TUM's frame
track.load_from_fastf1(raceline='data/racelines/monza.csv')                   # placed on the FastF1 session
```

**Which path to use for 2026:** `--use-tumftm` on the 8 rounds whose TUM layout is current (Shanghai, Suzuka, Montreal, Spielberg, Silverstone, Spa, Budapest, Monza; the `raceline` field in `config/events.py`). The other 7 (Melbourne, Barcelona, Miami, Monaco, Zandvoort, Madrid, Baku) have no current map of the driven line yet: only the FastF1 map line, whose curvature is unreliable.

`--use-tumftm` places the raceline on the session: it is registered onto the FastF1 line (rotation and shift), takes that line's height, starts at its timing line, and gets the FIA Straight Mode zones. A raceline more than 15 m from the session's layout anywhere is refused: 2026 Melbourne and Barcelona have changed since the TUM data were made.

- **Pros**: Smoother curvature, theoretically optimal racing line
- **Cons**: May not match actual F1 racing line (different constraints)
- **Best for**: Pure optimization studies, comparing strategies

### Calibration (2026 qualifying, work in progress)

`calibration/` fits one set of car parameters (drag and downforce areas, aero balance, Straight Mode drag, tyre grip scale, ICE power) to the 2026 pole laps of the 8 rounds with a current raceline:

```python
from calibration.dataset import reference_lap          # pole lap on the model's path, cached in data/cache/calibration
from calibration.fit import Fit, report, print_report
fit = Fit([2, 5, 9, 10, 13], ["c_w_a", "c_z_a", "mu_scale"])   # train rounds, parameters to fit
params = fit.run(max_evaluations=15)                   # least squares on the speed trace and lap time
print_report(report(params, [3, 8, 11]))               # held-out rounds
```

A round's residuals are the speed error at every measured sample (weighted like 100 samples, 8 km/h scale) and the lap-time error (0.2 s scale). Jacobians are forward differences, with the solves run in parallel; a failed solve rejects the step instead of ending the fit.

- **Curvature bound** (`calibration/practice.py`): the placed TUM lines are tighter than the driven line at a few fast corners, so the curvature is capped where the speeds of the sessions before qualifying (practice, sprint qualifying) would need more than 5 g. Qualifying data is never used, so held-out rounds stay predictions.
- **Joint fit** (`calibration/joint.py`): the shared parameters plus one aero level per track (ClA × (1 + k), CdA × (1 + 0.5 k)), each level set from that event's fastest practice lap under its session's energy rules; only the training rounds' pole laps set the shared parameters.
- **Diagnostics** (`calibration/plots.py`): model and measured speed traces per round.

**Status:** the targets (held-out lap time median ≤ 0.5 %, max ≤ 1 %) are not met yet. Best so far: median 1.30 %, max 1.71 % with one shared car; the joint fit gives 1.43 % and 2.45 %. The errors follow track type (too fast at Shanghai and Budapest, too slow at Monza and Spa), which points at how the model spends energy on the straights.

### Critical: SOC Boundaries Affect Results Dramatically

The initial and final State of Charge constraints fundamentally change the optimization:

| Strategy | Initial SOC | Final SOC | Behavior |
|----------|-------------|-----------|----------|
| Qualifying | 90% | 20% | Maximum deployment, don't save energy |
| Race (aggressive) | 50% | 30% | Deploy more than harvest |
| Race (conservative) | 50% | 50% | Charge-sustaining, balanced |
| Undercut prep | 50% | 80% | Harvest mode, prepare for attack |

```bash
# Qualifying setup (use all energy)
python main.py --track Monaco --initial-soc 0.9 --final-soc-min 0.2 --plot

# Conservative race strategy
python main.py --track Monaco --initial-soc 0.5 --final-soc-min 0.5 --plot
```

---

## Baseline Strategies

Compare the optimal NLP solution against rule-based strategies:

```bash
python scripts/compare_baselines.py
```

### Available Strategies

| Strategy | Description | Typical Gap to Optimal |
|----------|-------------|------------------------|
| **Offline Optimal** | NLP-computed globally optimal trajectory | Reference |
| **Optimal Tracking** | Follows NLP trajectory with feedback control | +0.05–0.1s |
| **Smart Heuristic** | Deploy on straights when grip available, harvest when braking | +0.3–0.8s |
| **Target SOC** | Proportional deployment based on SOC deviation from target | +0.5–1.0s |
| **Greedy (KERS)** | Deploy during hard acceleration, harvest during braking | +0.8–1.5s |
| **Always Deploy** | Deploy whenever accelerating and have charge | +1.0–2.0s |

### Why Are Baselines Slower Even With ERS?

A critical insight from this project:

> **Baselines with ERS are often slower than the optimal solution without additional ERS power.**

This happens because:

1. **Grip-Limited Acceleration**: Deploying ERS power in corners where tires are already at the grip limit provides **zero benefit** and wastes energy. The optimal solver knows exactly where grip headroom exists.

2. **Velocity Profile Mismatch**: Baseline strategies track a reference velocity but use feedback control with inherent lag. The optimal solution jointly optimizes velocity and energy.

3. **Energy Timing**: Deploying ERS 50m too early or too late on a straight can cost 0.1s. The NLP precisely times deployment to maximize velocity gain per joule.

4. **Cumulative Errors**: Small suboptimalities compound. A baseline making 20 "almost right" decisions loses to the globally optimal solution.

---

## Generating Poster Plots

Generate publication-quality visualizations:

```bash
# Generate all individual poster plots
python visualization/poster_plots.py

# Output saved to: figures/poster_plots/
```

### Available Poster Plots

| Plot | Filename | Description |
|------|----------|-------------|
| Track ERS Layout | `track_{name}_ers_layout.png` | Track map colored by ERS power |
| SOC Strategies | `soc_strategies_overlapped.png` | Compare different SOC constraints |
| Regulation Velocity | `regulation_velocity_comparison.png` | 2025 vs 2026 speed profiles |
| Regulation ERS | `regulation_ers_comparison.png` | Power deployment comparison |
| Lap Time Bars | `soc_strategies_laptime_comparison.png` | Bar chart of strategy performance |

### Driver Speed Profile Comparison

```bash
# Compare multiple drivers
python visualization/plot_driver_speed_profile.py \
    --track Monaco --year 2024 \
    --drivers VER LEC NOR \
    --save monaco_drivers_comparison.png
```

---

## Results & Outputs

### Output Directory Structure

Results are organized by track and timestamp:

```
results/
└── monaco/
    └── 20240115_143022/
        ├── summary.txt              # Human-readable results summary
        ├── data/
        │   ├── results_summary.json # Complete results as JSON
        │   ├── distance.npy         # Numpy arrays for analysis
        │   ├── velocity_optimal.npy
        │   ├── velocity_no_ers.npy
        │   ├── velocity_with_ers.npy
        │   ├── soc_optimal.npy
        │   ├── ers_power.npy
        │   ├── throttle.npy
        │   └── brake.npy
        └── plots/
            ├── 01_track_analysis.png
            ├── 02_offline_solution.png
            ├── 03_ers_comparison.png
            ├── 04_simple_results.png
            └── 05_lap_animation.gif  # If --save-animation
```

### Key Metrics

The solver reports:

- **Lap Times**: No ERS / With ERS / Optimal
- **Time Improvement**: Seconds gained vs. baseline
- **Energy Usage**: Deployed / Recovered / Net (MJ)
- **SOC Trajectory**: Initial → Final, swing range
- **Solver Performance**: Convergence status, solve time

### Example Output

```
========================================================================
F1 ERS OPTIMIZATION RESULTS - MONACO
========================================================================
Regulations Year:      2025

TRACK INFORMATION:
Track:                 Monaco
Total Length:          3337 m
Number of Segments:    668

LAP TIME PERFORMANCE:
Lap Time (No ERS):      75.432 s
Lap Time (With ERS):    74.891 s
Lap Time (Optimal):     74.956 s

Improvement vs No ERS:  0.476 s (0.63%)
Gap to Theoretical:     0.065 s

ENERGY MANAGEMENT:
Initial SOC:            50.0%
Final SOC:              32.1%
Energy Deployed:        2.847 MJ
Energy Recovered:       1.923 MJ
Net Energy Used:        0.924 MJ
========================================================================
```

---

## Project Structure

```
f1-ers-optimal-control/
├── main.py                 # Entry point
├── calibration/            # Phase 4: reference laps, parameter fit, diagnostics
├── config/
│   ├── __init__.py
│   ├── ers.py              # ERS regulations (2025/2026)
│   ├── events.py           # 2026 per-event energy rules and Straight Mode zones
│   └── vehicle.py          # Vehicle parameters, tire model
├── models/
│   ├── __init__.py
│   ├── car.py              # Car model shared by the solvers
│   ├── geometry.py         # Closed 3D line fitted to pooled laps
│   ├── telemetry.py        # FastF1 events, clean laps, cached geometry, Straight Mode zones
│   ├── track.py            # Track loading (FastF1/TUMFTM)
│   └── vehicle_dynamics.py # Physics model (CasADi)
├── solvers/
│   ├── __init__.py
│   ├── base.py             # Base classes, OptimalTrajectory
│   ├── forward_backward.py # Grip-limited velocity solver
│   └── spatial_nlp.py      # CasADi/IPOPT optimization
├── strategies/
│   ├── __init__.py
│   ├── base.py             # BaseStrategy interface
│   └── baselines.py        # Rule-based strategies
├── simulation/
│   ├── __init__.py
│   └── lap.py              # Time-domain simulation
├── visualization/
│   ├── __init__.py
│   ├── track_viz.py
│   ├── results_viz.py
│   ├── animation.py
│   ├── poster_plots.py
│   └── plot_driver_speed_profile.py
├── scripts/
│   └── compare_baselines.py
├── utils/
│   ├── __init__.py
│   └── run_manager.py      # Results organization
├── data/
│   ├── cache/              # FastF1 cache (created on first run)
│   └── racelines/          # TUMFTM racelines (optional)
├── results/                # Output directory
├── figures/                # Generated plots
├── docs/
│   └── poster.png          # Project poster
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Technical Details

### Vehicle Model

Based on [TUMFTM's laptime-simulation](https://github.com/TUMFTM/laptime-simulation):

| Parameter | 2025 Value | 2026 Value | Unit |
|-----------|------------|------------|------|
| Mass (with driver) | 798 | 768 | kg |
| ICE Power | 575 | 400 | kW |
| MGU-K Power | 120 | 350 | kW |
| Deployment Limit | 4.0 | Unlimited | MJ/lap |
| Recovery Limit | 2.0 | 8.5 | MJ/lap |
| Battery Capacity | 4.5 | 4.5 | MJ |

### Tire Model

A friction ellipse per axle, (F_x/F_x,max)² + (F_y/F_y,max)² ≤ 1, with load-dependent coefficients per tyre.
Axle loads include downforce and longitudinal load transfer; the axles share the lateral force by the steady
yaw balance; traction is on the rear axle only, braking on both:

```
μ(Fz) = μ₀ + (dμ/dFz) × (Fz - Fz₀)
```

| Parameter | Front | Rear |
|-----------|-------|------|
| μx (longitudinal) | 1.65 | 1.95 |
| μy (lateral) | 1.85 | 2.15 |
| dμ/dFz | -5×10⁻⁵ | -5×10⁻⁵ |

### Solver Configuration

The spatial NLP uses:
- **Transcription**: Direct collocation (selectable order)
- **NLP Solver**: IPOPT with MA97 or MUMPS linear solver
- **Discretization**: 5m spatial steps (typically 500-1000 nodes per lap)
- **Tolerances**: 1×10⁻⁴ (optimal), 1×10⁻³ (acceptable)

#### Collocation Methods

| Method | Order | Formula | Use Case |
|--------|-------|---------|----------|
| **Euler** | 1st | `x[k+1] = x[k] + h·f(x[k])` | Fast prototyping, coarse solutions |
| **Trapezoidal** | 2nd | `x[k+1] = x[k] + (h/2)·(f[k] + f[k+1])` | Good balance of speed/accuracy |
| **Hermite-Simpson** | 4th | Simpson's rule with Hermite midpoint | High accuracy, publication quality |

The Hermite-Simpson method uses the separated form:
```
x_mid = (x[k] + x[k+1])/2 + (h/8)·(f[k] - f[k+1])     # Hermite interpolation
x[k+1] = x[k] + (h/6)·(f[k] + 4·f_mid + f[k+1])       # Simpson quadrature
```

**Recommendation**: Use `--collocation trapezoidal` for general use, `--collocation hermite_simpson` for final results or when comparing against real telemetry.

---

## Roadmap

### Modeling Fidelity

- [ ] **Dynamic Tire Model**: Upgrade the static "friction circle with load-dependent coefficients" to include thermal degradation and wear factors for multi-lap accuracy.
- [x] **3D Track Geometry**: elevation (gradient and vertical curvature) from pooled telemetry. Banking is not modelled.

### Analysis & Validation

- [ ] **Sensitivity Analysis**: Create a script to sweep parameters (mass, drag, grip) and plot their impact on lap time and optimal ERS usage.
- [ ] **Validation Overlays**: Add an automated plot comparing the "Optimal" velocity profile directly against the loaded "FastF1" telemetry to visually quantify.

### Tooling & Reproducibility

- [ ] **Docker Support**: Containerize the environment to standardize the installation of `uv`, `ipopt`, and linear solvers (MA97/MUMPS) across different operating systems.

### Strategy & Learning

- [ ] **Race Strategy Solver**: Expand the "Multi-lap stint optimization" to include stint-level tire-compound choices.
- [ ] **RL**: hybrid residual model-based RL

---


## References

### Academic Sources

1. **Limebeer & Perantoni** (2014) - "Optimal control for a Formula One car with ERS"
2. **TUMFTM** - [laptime-simulation](https://github.com/TUMFTM/laptime-simulation), [trajectory_planning_helpers](https://github.com/TUMFTM/trajectory_planning_helpers)
3. **Dal Bianco et al.** - Quasi-steady-state lap time simulation
4. **Heilmeier et al.** (2020) - "Application of Monte Carlo Methods to Race Simulation"

### Validation

The implementation follows methodologies validated against:
- Real driver telemetry (within 0.1-0.5s)
- Published F1 results
- TUMFTM race simulation results

---


<p align="center">
  <em>🏁</em>
</p>
