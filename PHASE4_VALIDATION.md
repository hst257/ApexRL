# Phase 4 Validation

Recorded on Windows, Python 3.11.8, with the project's CPU PyTorch virtual environment. Phase 4 keeps vehicle physics, tracks, and observation/action versions unchanged. Model files, plots, and machine-readable reports are local, ignored artifacts.

## Automated verification

- Full RL environment: **100 tests passed**, with one existing Pygame `pkg_resources` deprecation warning.
- Base Python without optional RL dependencies: **65 passed, 4 skipped**.
- After adding per-circuit TensorBoard metrics: **14 targeted training, bootstrap, and garage tests passed**.
- Compilation, dependency consistency, and `git diff --check` passed.

New checks cover balanced checkpoint selection, preference for broader validated drivers, bounded teacher actions, actor fitting without critic changes, and accurate separation of demonstrations from PPO transitions. Existing checks cover manual gameplay, deterministic simulation, lap exploits, optional imports, checkpoint compatibility, Windows spawning, publication, and playback isolation.

## Training experiments

| Run | Training | Scheduled result |
|---|---|---|
| `phase4-all-baseline` | 401,408 mixed-circuit PPO transitions from the published Rookie policy | Final policy: no valid laps on any circuit; progress 50.9%, 27.1%, 22.4% |
| `phase4-bootstrap` | 5 rounds, 116,562 teacher-labelled transitions, 12,500 actor updates | Every round: 3/3 valid laps on each circuit |
| `phase4-all-refine` | 53,248 PPO transitions from bootstrap round 1 | Final policy: 3/3 valid laps on each circuit |

The initial published Rookie policy completed 5/5 fresh Rookie laps and 0/5 on each harder circuit (`artifacts/phase4-baseline.json`, seed 800126). Pure mixed-circuit PPO did not solve those failures. Balanced checkpoint selection retained its starting policy rather than replacing it with the regressed final policy.

Demonstration training uses a conservative pure-pursuit and curvature/braking teacher. Later DAgger rounds also label learner-visited states. This teacher accesses track geometry during training. The game receives only the learned PPO actor, the existing 50 observations, and two actions; it has no teacher fallback. Demonstration transitions are not counted as PPO transitions.

Refinement used learning rate 0.00002, four environments, initial action standard deviation 0.03, and seed 168. The candidate frozen for publication is `runs/phase4-all-refine/checkpoints/ppo_30000.zip`, inherited from 23,425 demonstrations and 2,500 actor updates. It completed 3/3 scheduled laps per circuit before the independent publication check. The eventual scheduled best checkpoint was at 40,000 steps; publication evaluates the separately frozen 30,000-step candidate.

Recorded elapsed time, including evaluation: 729.69 seconds for the mixed PPO baseline and 369.46 seconds for refinement. Jobs overlapped on this computer; these are run durations, not isolated performance benchmarks.

## Publication and game playback

The frozen candidate passed the 90% gate on **20/20 fresh-seed valid laps per circuit** (60 total), starting with evaluation seed 900168. It is published locally as `models/all-circuits-v1.zip`, with matching metadata and evaluation sidecars. Its checksum matches the frozen candidate exactly.

| Circuit | Held-out valid laps | Best valid lap |
|---|---|---|
| Rookie Ring | 20/20 | 52.905 s |
| Apex Valley | 20/20 | 79.374 s |
| Velocity Circuit | 20/20 | 116.721 s |

An independent check loaded the **published copy** using fresh seed 1100168 and completed another **5/5 valid laps per circuit**. That gives 25/25 on each circuit, **75/75 total**, across publication and confirmation. The confirmation report is `runs/phase4-all-refine/published-confirmation.json`. These bounded checks use the configured ±0.3 m lateral and ±0.03 rad heading variation at the grid.

The identical candidate weights also completed a valid canonical lap on every circuit through the native Pygame application's `AISession`: 52.951 s, 79.441 s, and 116.822 s. Assertions verified that human lap records remained unchanged. This was scripted application driving, not human playtesting. `artifacts/phase4-game-verification.json` records those checks.

The native Garage was launched after publication. Its preferred entry is the new all-circuit driver, showing 30,000 PPO steps, 23,425 demonstrations, and 20/20 circuit validation. `artifacts/phase4-garage.png` and all three `phase4-<track>-lap.png` captures document the presentation; the Garage and Velocity lap captures were visually inspected for readable, unobstructed results.

## Evidence and limits

Run directories contain compatible checkpoint sidecars, seeded episode records, and evaluations. PPO runs also include TensorBoard logs, `learning_curves.png`, and `evaluation_trajectories.png`. The refinement plots were inspected and show complete trajectories on all circuits.

Scheduled seeds selected candidates and do not count as independent publication evidence. Competence here means valid single laps from the configured starting-grid variation; it does not establish racing against opponents, unseen circuits, aggressive lap-time optimization, or human driving feel. The earlier Rookie specialist remains available for faster Rookie Ring laps. Source changes and model outputs remain local.
