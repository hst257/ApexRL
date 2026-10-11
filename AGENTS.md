Whatever action you can do yourself, Please do yourself, this includes starting apps and verification

# Repository Guidelines

## Project Status & Scope

ApexRL targets Windows and Python 3.11+. Phase 1 implements gameplay; Phase 2 adds optional Gymnasium, PPO, evaluation, and policy playback.

Preserve manual gameplay and keep RL dependencies optional. Do not begin subsequent phases without explicit authorization.

## Project Structure & Module Organization

- `main.py`: game entry.
- `apexrl/core/`: vehicle dynamics, geometry, surfaces, collisions, checkpoints, timing, and headless simulation.
- `apexrl/game/`: Pygame application, input, camera, rendering, HUD, menus, effects, and audio.
- `apexrl/utils/`: settings and persistence.
- `apexrl/rl/`: environment, observations, rewards, checkpoints, evaluation, training, and spectator rendering.
- `config/`: JSON game and vehicle parameters.
- `tracks/`: original Rookie Ring, Apex Valley, and Velocity Circuit definitions.
- Graphics and audio are generated in `apexrl/game/`.
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
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
.\.venv\Scripts\python.exe train.py --smoke
```

No separate build is required. `visual_check.py` captures screens. `evaluate.py --model <zip>` measures policies; `watch.py --model <zip>` launches playback. Store outputs in ignored `runs/` or `artifacts/`. See `PHASE2.md` for CPU installation.

## Coding Style & Architecture

Use four-space indentation, type hints for public interfaces, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. No formatter or linter is configured yet.

Keep tuning in dataclasses or JSON. Use SI units and clockwise-positive radians. Physics uses a fixed 1/120-second timestep.

Core modules must not import Pygame or read wall-clock time. Controllers produce `VehicleControl`; renderers only read state. Seed stochastic behavior explicitly. Bump RL observation/action versions when their meaning or normalization changes.

## Testing Guidelines

Name suites `test_*.py` and cases `test_<behavior>`. Cover physics, tracks, collisions, lap exploits, determinism, Gymnasium contracts, reward exploits, termination/truncation, checkpoint reload, and Windows spawning. RL tests skip when optional dependencies are absent. No coverage percentage is configured.

## Commit & Pull Request Guidelines

The initial commit uses `Version 1: Game Build`. Prefer clear imperative subjects, such as `Add seeded PPO evaluation`, or descriptive milestone subjects.

PRs should explain changes, link applicable issues, report tests and benchmarks, and include visual screenshots. Distinguish screenshot review from driving verification. Exclude environments, caches, generated saves, and unlicensed assets.
