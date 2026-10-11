Whatever action you can do yourself, Please do yourself, this includes starting apps and verification

# Repository Guidelines

## Project Status & Scope

ApexRL is a Python 3.11+ top-down Formula-style racing game targeting Windows. Phase 1 implements a standalone time-trial game and headless simulation. There is no Git history in this workspace; create modules only when needed.

Implement the standalone game first. Do not add Gymnasium, PyTorch, PPO, rewards, AI drivers, training code, or future-phase placeholders without explicit authorization.

## Project Structure & Module Organization

- `main.py`: game entry point.
- `apexrl/core/`: vehicle dynamics, geometry, surfaces, collisions, checkpoints, timing, and headless simulation.
- `apexrl/game/`: Pygame application, input, camera, rendering, HUD, menus, effects, and audio.
- `apexrl/utils/`: settings and persistence.
- `config/`: JSON game and vehicle parameters.
- `tracks/`: original Rookie Ring, Apex Valley, and Velocity Circuit definitions.
- Graphics and audio are generated in `apexrl/game/`; system fonts have Pygame fallback.
- `tests/`: Pytest suites; `saves/`: generated settings and lap records.

## Build, Test, and Development Commands

Use `python` on Windows from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest tests/test_physics.py -q
.\.venv\Scripts\python.exe benchmark.py --output artifacts/benchmark.json
```

These create an environment, install dependencies, launch, test, and benchmark 100,000 headless steps per track. No separate build is required. `python visual_check.py` produces scripted screenshots under `artifacts/visual/`.

## Coding Style & Architecture

Use four-space indentation, type hints for public interfaces, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. No formatter or linter is configured yet.

Keep tuning in dataclasses or JSON. Use SI units; headings are radians, with zero pointing right and positive rotation clockwise. Physics uses a fixed 1/120-second timestep.

Core modules must not import Pygame or depend on wall-clock time. Controllers produce continuous `VehicleControl` actions; renderers only read simulation state. Avoid mutable global state and seed stochastic behavior explicitly.

## Testing Guidelines

Name suites `test_*.py` and cases `test_<behavior>`. Cover physics stability, track validity, collision footprints, ordered forward checkpoint crossings, recovery, determinism, and headless operation. No coverage percentage is configured. Add regression tests for simulation bugs.

## Commit & Pull Request Guidelines

No existing commit convention is available. Use concise imperative subjects, such as `Add ordered checkpoint validation`.

PRs should explain behavior changes, reference applicable issues, list actual test and benchmark results, and include screenshots for visual changes. Distinguish screenshot review from hands-on driving verification. Exclude virtual environments, caches, generated saves, and unlicensed assets.
