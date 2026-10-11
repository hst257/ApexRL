# APEX RL

An original top-down Formula-style time-trial racer for Windows, built with Python and Pygame. Phase 1 includes the playable standalone game and a deterministic, graphics-independent simulation. **No reinforcement learning is implemented or installed.**

## Setup and launch

Requires Python **3.11+**. From this repository in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Using the virtual environment's executable avoids PowerShell activation-policy issues. If the project dependencies are already installed, `python main.py` is sufficient. No separate build step is required. Run from a local checkout for best disk performance.

To launch directly into a circuit:

```powershell
python main.py --track apex_valley
```

## Driving

| Input | Action |
|---|---|
| W / Up | Accelerate |
| S / Down | Brake to a stop; no reverse gear |
| A / Left, D / Right | Smoothed left/right steering |
| R | Recover to the last safe checkpoint; invalidate this lap |
| Escape | Pause/resume; go back in menus |
| Tab | Toggle expanded circuit overview |
| F3 | Toggle telemetry and geometry overlays |
| Menu arrows + Enter | Navigate/select |
| Settings Left / Right | Decrease/increase the selected setting |
| Mouse | Hover/select menu buttons; click settings to cycle |

The car waits through a three-second countdown. Brake before tight corners and use short steering inputs at speed. Tire forces saturate when grip is exceeded; release steering and reduce speed to recover a slide. Grass reduces traction, engine effectiveness, and coasting distance. Solid barriers damp impacts and separate the car from contact.

Race sessions have unlimited laps. All checkpoint gates must be crossed in order and forward, followed by the finish. Repeated finish crossings and reverse driving cannot set records. Two continuous seconds with all wheels off asphalt/curbs invalidates the lap; eight seconds triggers recovery. R also invalidates the lap without resetting its timer. The next forward finish crossing begins a fresh attempt. Restart clears the session and countdown, preserving saved records.

The HUD shows attempt number, valid completed laps, simulation-based timing, previous valid lap, persistent personal best, speed, and a circuit-derived minimap. Pause, loss of window focus, menus, and settings freeze race time.

## Circuits and presentation

- **Rookie Ring:** approximately 1.20 km, 16 m road, forgiving curves and a moderate hairpin.
- **Apex Valley:** approximately 1.80 km, 13 m road, mixed-radius corners and a chicane.
- **Velocity Circuit:** approximately 2.50 km, 10–13 m road, sweepers, hairpins, and switchbacks.

Track lengths are measured from the derived geometry. Each circuit has an original environment palette, curbs, barriers, trackside trees, garages, grandstands, signs, and start-grid markings. Sprites, scenery, and synthesized audio are generated locally; no third-party racing artwork is downloaded. Fonts use Windows system typography with Pygame fallback.

The default resizable window is 1280×720. Rendering targets 60 FPS, with aspect-preserving scaling and letterboxing on differently shaped windows. Settings include master volume, steering sensitivity, camera response, detail level, resolution, and fullscreen. Low detail reduces scenery texture and disables effects without changing physics. An unavailable audio device does not prevent play.

## Architecture and simulation API

- `apexrl/core/`: continuous controls, vehicle configuration/state, bicycle-model physics, spline geometry, spatial track queries, footprint collision checks, lap timing, and simulation snapshots.
- `apexrl/game/`: application state transitions, keyboard adapter, camera, original graphics, menus, bounded effects, and synthesized audio.
- `apexrl/utils/`: validated preferences and atomic versioned JSON persistence.
- `config/`, `tracks/`: tunable JSON parameters and circuit definitions.
- `tests/`: behavioral, geometry, determinism, persistence, and application tests.

Core code never imports Pygame or reads a clock. Rendering consumes immutable snapshots. Multiple simulation instances can share immutable track arrays while maintaining independent vehicle and race state.

```python
from apexrl.core import RacingSimulation, VehicleControl, VehicleConfig, load_tracks
from apexrl.core.simulation import SimulationConfig

track = load_tracks("tracks")[0]
simulation = RacingSimulation(
    track,
    VehicleConfig.load("config/vehicle.json"),
    seed=42,
    simulation_config=SimulationConfig(countdown_seconds=0),
)
simulation.reset()
for _ in range(1000):
    result = simulation.step(VehicleControl(steering=0.1, throttle=0.8), dt=1/120)
print(result.state.vehicle.speed)  # m/s
```

`step()` returns `StepResult(state, events)`; events identify checkpoints, collisions, recoveries, invalidations, completed laps, and session bests. `reset()` returns a fresh snapshot. `recover_car()` preserves race timing and invalidates the active lap. Core best times are session-local; the application maintains disk records separately.

Coordinates use meters, with X right and Y down. Heading zero points right; positive headings and steering rotate clockwise. Speeds and accelerations use SI units, with km/h conversion only in the HUD. The fixed interactive timestep is 1/120 second; API timesteps must be finite and in `(0, 1/60]`. Controls clamp finite values to steering `[-1, 1]` and throttle/brake `[0, 1]`; nonfinite values are rejected before updates.

Track definitions use cyclic centripetal Catmull–Rom control points resampled at approximately two meters. Width profiles, finite gates, arc lengths, tangents, curvature, curb bands, and barrier contours are derived once. A uniform spatial grid accelerates exact nearest-segment projection with a global fallback. Geometry is validated at load time.

The reduced dynamic bicycle model blends into kinematic low-speed behavior, limits tire forces by grip and longitudinal demand, and uses semi-implicit integration. Contact checks sweep the physical oriented car footprint. The application interpolates snapshots and caps catch-up at 16 steps per frame. Excess stalled-frame backlog is discarded, so simulation time can advance slower than wall time during heavy stalls; physics itself does not change with render frequency.

## Configuration and saves

Edit `config/vehicle.json` for vehicle tuning and `config/game.json` for application defaults and race durations. Circuits are defined in `tracks/*.json`. All three must pass geometry validation to launch.

Settings and per-track personal bests are written under `saves/`. Save files have a versioned envelope and are replaced atomically. Corrupt saves use defaults with a visible warning; failed writes leave gameplay and in-memory state usable. Records are keyed by track-definition hash, vehicle configuration, race-recovery/shortcut thresholds, and physics rules version, preventing incompatible tuning from sharing records. Delete generated save files to reset preferences or records. Generated saves and validation artifacts are excluded from version control.

## Tests and measured validation

The recorded Phase 1 run passed **58 tests** and measured **3,592.9–3,902.0 complete headless steps/second**, with 100,000 measured steps on each track. See [VALIDATION.md](VALIDATION.md) for per-track measurements, native screenshot evidence, and the distinction between automated checks and unverified human driving feel.

```powershell
python -m pytest -q
python -m pytest tests/test_physics.py -q
python benchmark.py --steps 100000 --output artifacts/benchmark.json
python visual_check.py
```

The benchmark performs 1,000 warm-up steps and at least 100,000 measured complete simulation steps per circuit with a reproducible mixed-control sequence. It does not create a Pygame window and reports actual elapsed time and throughput. The sequence exercises off-track driving, contacts, and recovery; it is not an AI driver. Results depend on machine, Python, and system load.

Tests cover acceleration, braking, drag, saturation, slides, valid/invalid geometry, spatial queries, finite gate crossings, lap exploits, recovery, collision sweeps, seeded independent instances, import/stepping with Pygame blocked, corrupt saves, input navigation, pause, resizing, and different render schedules. Pytest temporary files live under `artifacts/pytest-temp` because the execution sandbox blocks atomic renames in its system temporary directory.

`visual_check.py` saves presentation screenshots and a render-only throughput report to `artifacts/visual/`, using SDL test drivers by default. `python visual_check.py --native` uses the desktop display. Render-only throughput excludes simulation and is not an interactive-game FPS claim. A bounded native launch is also available:

```powershell
python main.py --track rookie_ring --frames 120 --screenshot artifacts/native-race.png
```

## Phase 1 limits and next phase

Handling is simcade, not a professional tire or aerodynamics simulation. Grass/curb effects average wheel contacts; contact resolution is forgiving rather than a rigid-body solver. There is no reverse gear, sector timing, gravel, multiplayer, gamepad input, or replay-file UI. Visual inspection and scripted input tests do not establish hands-on driving feel; human playtesting remains necessary.

After explicit Phase 2 approval, the existing headless action/state API can support a Gymnasium adapter, numerical observations, reward and episode rules, reproducible evaluation, and training dependencies. None of those are part of Phase 1.
