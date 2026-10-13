# ApexRL 1.0 Release Validation

Recorded on **2026-10-13** on Windows x64, Python 3.11.8, Pygame 2.6.1, and CPU PyTorch 2.14.1. Version 1.0 preserves core physics and the observation/action contracts, so the validated drivers remain compatible.

## Source and gameplay

- **106 tests passed**, including controls, physics, deterministic stepping, lap exploits, collisions, saves, Windows spawning, PPO reload, publication, and manual play without RL imports.
- New checks cover responsive driver loading, cancellation, loading failures, focus loss during loading, stable driver selection after refresh, published-only browsing, and frozen resource/save paths.
- Dependency consistency and whitespace checks passed. The only test warning was Pygame's existing `pkg_resources` deprecation.
- Native Windows captures covered the main menu, all tracks, controls, settings, countdowns, racing, maps, telemetry, pause, and a resized window.
- A native asynchronous-loading check rendered **258 frames** while preparing a driver, cancelled the first load, reopened it, and completed a valid Rookie lap in **52.951 seconds**. The Garage, loading screen, and track selection captures were visually inspected.

AI engine audio now follows the AI vehicle and stops when paused or finished. AI laps retain separate session timing and cannot update human personal bests. Published drivers are the default Garage list; training runs remain available behind a toggle.

## Measured performance

| Circuit | Complete physics steps | Elapsed seconds | Steps/second |
|---|---:|---:|---:|
| Rookie Ring | 100,000 | 56.654 | 1,765.1 |
| Apex Valley | 100,000 | 53.913 | 1,854.8 |
| Velocity Circuit | 100,000 | 57.522 | 1,738.5 |

These measurements overlapped packaging and tests; they are not isolated performance comparisons with earlier phases. Native render-only checks measured 163.8–174.0 FPS, excluding physics and AI inference. Neither measurement guarantees interactive FPS on another machine.

## Drivers and distribution

The release contains the existing immutable `all-circuits-v1` and `rookie-reliable-v1` checkpoint pairs and evaluation reports. Their previous held-out results are 75/75 valid all-circuit laps and 50/50 valid Rookie specialist laps. The all-circuit policy is conservative; the specialist is faster on Rookie Ring. It is not validated on the harder circuits.

The Windows package is an on-directory PyInstaller build containing the interpreter, CPU neural runtime, game resources, both drivers, app icon, and dependency notices. The matching source ZIP includes tests and training tools. User saves are excluded from archives and kept beside the executable. The builder preserves existing portable saves when rebuilding.

Packaged checks use `--self-check` to exercise manual acceleration/pause on all circuits and valid laps for every published driver/circuit pairing. This checks the actual executable and bundled resource/checkpoint loading rather than treating a successful build as sufficient evidence.

The executable passed all three manual checks and all four published driver/circuit pairings, with human records unchanged. Recorded valid laps were **52.951 s**, **79.441 s**, and **116.822 s** for the all-circuit driver, plus **29.648 s** for the Rookie specialist. Packaging checks caught and corrected a PyTorch support-module exclusion and a missing Stable-Baselines version resource. The build now runs this verification gate before creating archives.

## Limits

Verification uses scripted controls and deterministic policies; it is not hands-on human playtesting. There is no multiplayer, AI opponent interaction, gamepad support, or professional tire model. The package targets Windows x64. Training remains a source workflow, and the standalone game keeps neural imports lazy during manual play.

Evidence is stored in ignored `artifacts/release-*`, while training evidence remains under the Phase 3/4 run directories. Distribution files and checksums are under ignored `dist/`.
