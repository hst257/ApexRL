# Phase 3 Validation

Measured on 2026-10-11 using Windows, Python 3.11.8, CPU PyTorch 2.14.1+cpu, and the Phase 2 dependencies. All runs use the original circuits, vehicle physics, ordered checkpoints, and lap-validity rules.

**Latest result:** `models/rookie-reliable-v1.zip` passed its 20-episode publication check and a separate 30-episode confirmation check, completing **50/50 fresh Rookie Ring laps**. Its best measured held-out lap was **29.5682 seconds**. It is preferred in the AI Garage. These bounded checks establish performance on the configured Rookie Ring starts; they do not establish competence on the harder circuits or guarantee every possible start.

## Reliability run and published driver

`config/training-reliable.json` transfers the earlier 480,032-transition lap-completing policy into a fresh optimizer with learning rate 0.00005, entropy coefficient 0.0001, initial action standard deviation 0.05, four environments, and seed 84. It retains the original reward, episode settings, observations, and vehicle physics.

The run requested 120,000 new transitions and completed **122,880**, taking **398.42 seconds** including setup and evaluation under concurrent test/evaluation load. Periodic five-episode Rookie Ring validation passed 4/5 at 25,000 transitions and 5/5 at 50,000, 75,000, and 100,000. The final policy also passed 5/5 on Rookie Ring but 0/5 on each harder circuit. The published model is the immutable **50,000-transition checkpoint**, not the later final policy.

```powershell
.\.venv\Scripts\python.exe promote.py --model runs/phase3-rookie-reliable/checkpoints/ppo_50000.zip --name rookie-reliable-v1 --track rookie_ring --episodes 20 --seed 400084 --minimum-success .9
.\.venv\Scripts\python.exe evaluate.py --model models/rookie-reliable-v1.zip --track rookie_ring --episodes 30 --seed 600084 --output runs/phase3-rookie-reliable/published-confirmation.json
```

Both seed batches are distinct from training and scheduled validation. Publication completed **20/20** valid laps, best **29.5682 seconds**. Confirmation evaluated the frozen published copy and completed **30/30**, best **29.5733 seconds**. Every terminal result was `lap_completed`. The model, compatible metadata, and publication report are stored together under `models/`; confirmation remains under the originating run. The garage displays 20/20 publication laps for Rookie Ring and reports that other selected circuits have no publication result.

Native integrated playback of the published model completed a **29.648-second** canonical-start lap and held the result on screen. `artifacts/phase3-published-lap.png` and `artifacts/phase3-published-garage.png` were inspected. The AI header, publication statistics, inherited training count, progress, completed-lap banner, minimap, and controls were readable. The earlier model's evidence below is retained to explain why reliability training was needed.

The latest full suite passed **95 tests in 47.66 seconds**, with the existing Pygame dependency warning. After the final metadata guard change, the five garage tests passed in **4.42 seconds**, including malformed publication metadata. The game-only installation passed **64 tests with 3 RL modules skipped in 24.15 seconds**. TensorBoard was launched for all runs at `http://127.0.0.1:6007`; its scalar endpoint returned HTTP 200 and included the reliability run. The Phase 2 commit `027c2b5` was pushed to the approved GitHub `main` branch and independently verified with `git ls-remote`.

## Earlier verification

- Full RL installation: **94 tests passed in 39.60 seconds**. Pygame emitted its existing `pkg_resources` deprecation warning.
- Base installation without RL dependencies: **63 tests passed, 3 optional RL test modules skipped in 22.58 seconds**.
- After the final AI toast fix, all **11 application/garage tests passed in 11.98 seconds**. The final catalog label change was checked with a native garage launch.
- New tests cover metadata discovery, missing optional packages, import isolation, pause/focus-loss timing, restart, actual policy action equality after loading, terminal-step freezing, human record isolation during an AI lap, reward-only tuning with identical physics/observations, transfer versus resume, publication thresholds, report/model identity, and overwrite/path protection.
- Compilation, `pip check`, and `git diff --check` passed. Git separately reported its configured LF-to-CRLF conversion notices.

## Longer PPO training

| Run | Requested new transitions | Actual new transitions | Final policy transition count | Result |
|---|---:|---:|---:|---|
| `phase3-rookie-baseline` | 200,000 | 200,704 | 200,704 | Final Rookie Ring progress 43.61%, 0/3 valid laps |
| `phase3-rookie-refine` | 300,000 | 300,032 | 480,032 | Final Rookie Ring evaluation completed 3/3 valid laps |

The baseline took **296.14 seconds**, including setup and evaluations. Refinement took **518.05 seconds**, including evaluations and competing workloads. It resumed the baseline's best checkpoint at 180,000 transitions; it did not resume the baseline's later final policy. Learning was inconsistent, so periodic best checkpoints were retained. The final refinement update produced the selected best model.

The final evaluation used three seeded starts per circuit:

| Circuit | Valid laps | Mean maximum progress | Best valid lap |
|---|---:|---:|---:|
| Rookie Ring | 3/3 | approximately 100% | 29.247 seconds |
| Apex Valley | 0/3 | 26.07% | None |
| Velocity Circuit | 0/3 | 31.15% | None |

The harder-circuit failures were collisions. No generalization or competence on those circuits is claimed.

## Independent publication evaluation

```powershell
.\.venv\Scripts\python.exe promote.py --model runs/phase3-rookie-refine/best_model.zip --name rookie-v1 --track rookie_ring --episodes 10 --seed 200042 --minimum-success .9
```

This evaluated ten fresh jittered starts, distinct from the scheduled evaluation seeds. The selected checkpoint completed **7/10 valid laps**, with a best valid time of **29.2189 seconds**. Two episodes ended off-track and one ended in a collision. Mean maximum progress was **82.23%**; mean return was **181.86**.

The policy failed the required 90% success gate. The tool correctly exited with status 2, saved `runs/phase3-rookie-refine/promotion-rookie-v1.json`, and did **not** publish `models/rookie-v1.zip`. The earlier Phase 2 smoke policy also failed a five-episode check at 0% success and produced no published model.

This earlier checkpoint could complete valid Rookie Ring laps but remained inconsistent and was not published. The later reliability run above addressed these failures on fresh seed batches. Future tuning should continue using new validation seeds and broader starts to avoid optimizing to measured episodes.

## Corner-speed experiment

`config/training-corners.json` tested a curvature-based overspeed penalty, lower learning rate/entropy incentive, and reduced initial action noise after copying weights from the baseline's best policy. The transfer uses a fresh optimizer and separately records source training steps. The default Phase 2 reward remains unchanged.

The experiment requested 150,000 new transitions and completed **150,528** in **470.77 seconds**, including evaluation. Early deterministic progress increased to roughly 59%, but the final Rookie Ring policy stalled at **36.24% mean progress**, completing **0/3 valid laps**. It also completed no valid laps on the other circuits. This configuration did not outperform ordinary PPO refinement and remains explicitly experimental. Its reward values differ from the baseline, so raw returns should not be compared directly.

## Native launches and visual inspection

The garage and integrated AI playback were launched natively and their saved canvases inspected. A **2,100-frame native playback** of the selected policy completed a valid Rookie Ring lap in **29.247 seconds** and held the terminal result on screen. `artifacts/phase3-ai-lap.png` shows the completed lap, AI session best, previous lap, progress badge, result banner, and spectator controls without overlap. The garage screenshot is `artifacts/phase3-garage.png`.

Learning and trajectory plots were generated for all three runs; refinement learning curves were inspected visually. Simulation timing, pause/restart, and record separation have automated coverage. Hands-on driving feel, audible output, and every native input interaction were not human-tested. Generated models, logs, images, and plots remain ignored by Git. Phase 3 source changes remain local while the earlier Phase 2 commit is preserved separately.
