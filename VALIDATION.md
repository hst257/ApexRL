# Phase 1 Validation

Validated on 11 October 2026 using Python 3.11.8 and the native Windows SDL display driver. Machine-reported platform: `Windows-10-10.0.26200-SP0`; processor: `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel`.

## Automated checks

`python -m pytest -q`: **58 passed in 9.66 seconds**.

Coverage includes vehicle dynamics, stopping and speed limits, grip saturation and recovery, all three circuits, malformed geometry, locally folded barriers, exact spatial queries, checkpoint order, reverse/finish-line exploits, interpolated timing, recovery, independent seeded instances, operation with Pygame imports blocked, save failures, menus, focus-loss pause, resizing, input sensitivity, and render-schedule independence.

`python -m compileall -q apexrl main.py benchmark.py visual_check.py` completed successfully.

## Headless benchmark

Command: `python benchmark.py --output artifacts/benchmark.json`.

Each circuit received 1,000 warm-up steps followed by 100,000 measured complete simulation updates. No rendering or Pygame display initialization was involved. The deterministic mixed-control sequence includes contact/off-track/recovery behavior rather than an autonomous driver.

| Circuit | Measured steps | Elapsed seconds | Steps/second |
|---|---:|---:|---:|
| Rookie Ring | 100,000 | 27.833 | 3,592.9 |
| Apex Valley | 100,000 | 25.628 | 3,902.0 |
| Velocity Circuit | 100,000 | 26.083 | 3,833.9 |

These measurements are specific to this machine and run. Concurrent validation activity may affect throughput. Raw machine metadata and results are saved in `artifacts/benchmark.json`.

## Launch and visual inspection

- Launched the native application and completed `python main.py --track rookie_ring --frames 120 --screenshot artifacts/native-launch.png` successfully.
- Ran `python visual_check.py --native --output artifacts/native-visual`, capturing main/track/controls/settings menus, countdowns, racing, expanded maps, telemetry, pause, and a resized window.
- Inspected screenshots of all circuits and the key UI states. Verified readable layout, circuit-derived maps, camera framing, car visibility, curbs, start markings, and scenery. Corrected missing arrow glyphs, clipped trees at terrain chunk edges, translucent overview header interference, and tight-corner barrier folds.
- The native screenshot run measured cached render-only throughput of 175.6, 159.5, and 164.4 frames/second for the three circuits. These exclude simulation and are **not** measurements of interactive game FPS.

## Remaining acceptance limitations

Native computer-control tools were unavailable. Actual human keyboard driving, audio listening, and subjective handling satisfaction were **not verified**. Scripted continuous controls, keyboard translation tests, native launches, and screenshot inspection were performed. Full-circuit lap validation was tested with geometric trajectories; a human-driven valid lap remains a playtesting task.

Phase 1 functionality is implemented and automated checks pass. The requirement for independently confirmed satisfying handling remains unverified until human playtesting. The model is intentionally simcade, with forgiving footprint-based barrier response. Optional sectors, gravel, reverse gear, and gamepad controls are omitted. No reinforcement learning dependencies or functionality were added.
