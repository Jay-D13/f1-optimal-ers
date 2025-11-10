# F1 ERS Optimal Control – Quick Start (with **uv**)
---

## Prerequisites
- **Python** 3.10–3.12 (recommended: 3.11)
- **uv** (package & project manager by Astral)
- Internet access on first run (FastF1 will download session data)

> The code will save figures/animations such as `baseline_results.png`, `mpc_results.png`, `track_analysis.png`, and `mpc_lap_animation.gif` in the project folder.

---

## Install `uv`

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# then restart your shell or ensure ~/.local/bin is on PATH
```

### Windows (PowerShell)
```powershell
iwr -useb https://astral.sh/uv/install.ps1 | iex
```

Verify:
```bash
uv --version
```

---

## (Optional) Install a Python runtime with `uv`
If you don’t already have an appropriate Python available:
```bash
uv python install 3.11
```

---

## Create a virtual environment & install deps from `uv.lock`
From the project root (where the `uv.lock` lives):
```bash
uv sync
```

This will create a .venv environment to match `uv.lock` precisely.

> If you want to use the environment without `uv run`, activate it:
> - macOS/Linux: `source .venv/bin/activate`
> - Windows PowerShell: `.venv\Scripts\Activate.ps1`

---

## Set up FastF1 cache (required)
Create the cache directory once:
- macOS/Linux:
  ```bash
  mkdir -p data/cache
  ```
- Windows:
  ```powershell
  mkdir data\cache
  ```

---

## Run the code

```bash
uv run python main.py
```

On first run, FastF1 will download session telemetry (e.g., 2023 Monaco Q), so it can take a bit and will populate `data/cache/`.

---

## Outputs & where to look
- `track_analysis.png` – track visualization
- `baseline_results.png` – rule-based strategy results
- `mpc_results.png` – MPC strategy results (when MPC is available)
- `mpc_lap_animation.gif` – animated lap visualization (if creation succeeds)

--- 