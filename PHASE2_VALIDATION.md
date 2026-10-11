# Phase 2 Validation

Measured on 2026-10-11, Windows 10.0.26200, Python 3.11.8. Training used CPU PyTorch 2.14.1+cpu, Gymnasium 1.4.0, Stable-Baselines3 2.9.0, TensorBoard 2.21.0, and NumPy 2.4.6. Measurements vary with hardware and system load.

## Automated checks

- Optional-RL virtual environment: **86 tests passed in 22.65 seconds**. One Pygame dependency warning concerns deprecated `pkg_resources`.
- Base Python installation without RL packages: **59 tests passed, 2 RL modules skipped in 10.72 seconds**.
- Gymnasium and Stable-Baselines3 environment checkers passed.
- Contract checks cover 50-value observations, continuous actions, seed reproducibility, independent instances, termination versus truncation, invalid input rejection, reward/teleport rules, actual valid-lap events, and headless operation with Pygame and PyTorch imports blocked.
- Training checks verify changed neural-network weights, checkpoint reload, contract rejection, deterministic repeated evaluation, real Windows spawn workers, TensorBoard output, resume counts, short-run best-model creation, and protection of existing run directories.
- `pip check` reported no broken requirements; compilation and `git diff --check` passed. Git separately reported its configured LF-to-CRLF conversion notices.

## PPO smoke run and resume

```powershell
.\.venv\Scripts\python.exe train.py --smoke --run-dir runs/phase2-smoke
.\.venv\Scripts\python.exe evaluate.py --model runs/phase2-smoke/final_model.zip --episodes 2 --seed 100042 --output runs/phase2-smoke/reloaded-evaluation.json
.\.venv\Scripts\python.exe train.py --smoke --timesteps 1024 --resume runs/phase2-smoke/final_model.zip --run-dir runs/phase2-resume-smoke
```

The initial run collected **16,384 transitions** across two environments. Setup, learning, and evaluations took **25.17 seconds**. It saved periodic checkpoints at 8,192 and 16,384 transitions, best/final models with metadata, episode logs, TensorBoard events, and baseline/final evaluations. Reloaded evaluation reproduced the final report's episode records and summaries exactly. Resume added **1,024 transitions**, reaching **17,408**, and wrote results to a separate directory.

The untrained deterministic baseline stalled with zero progress on all three circuits. The final smoke policy accelerated but collided near the start. Its six held-out seeded episodes produced:

| Circuit | Mean maximum lap progress | Mean return | Valid laps | Terminations |
|---|---:|---:|---:|---|
| Rookie Ring | 1.92% | -8.19 | 0/2 | 2 collisions |
| Apex Valley | 1.35% | -8.10 | 0/2 | 2 collisions |
| Velocity Circuit | 1.08% | -7.87 | 0/2 | 2 collisions |

No valid lap time was inferred or recorded. This validates the training pipeline and limited policy change, not capable racing. Longer training, multiple seeds, and measured reward/hyperparameter tuning remain necessary. Generalization to the other two circuits has not been established.

## Headless RL throughput

`rl_benchmark.py` measured 10,000 agent transitions per circuit after warm-up, with four physics ticks per transition. Timed work includes observations, rewards, simulation updates, and episode resets. It excludes track preprocessing, policy inference, optimization, and rendering.

| Circuit | Elapsed seconds | Agent steps/second | Physics ticks/second |
|---|---:|---:|---:|
| Rookie Ring | 9.009 | 1,110.0 | 4,440.0 |
| Apex Valley | 8.386 | 1,192.5 | 4,769.9 |
| Velocity Circuit | 7.728 | 1,294.0 | 5,176.1 |

Each measured sequence performed one episode reset. These random-action measurements are not lap-completion or interactive-frame-rate claims. Raw results are in ignored `artifacts/rl-benchmark.json`. Historical core-only measurements remain in [VALIDATION.md](VALIDATION.md).

## Launch and visual checks

Native Pygame AI playback ran for 90 rendered agent steps on Rookie Ring, saved `artifacts/phase2-ai.png`, and exited normally. The saved canvas was inspected: car, road, timer, speed, minimap, AI progress badge, and spectator controls were readable. The screenshot represents three simulated seconds of the weak smoke policy. The manual game also launched natively for 60 frames and saved `artifacts/phase2-manual.png`.

Learning curves and measured evaluation trajectories were generated under `runs/phase2-smoke/` and visually inspected. TensorBoard started at `http://127.0.0.1:6006`; its scalar endpoint returned HTTP 200 with recorded rollout, optimization, and evaluation metrics. The local server was left running for inspection.

Automated launches and canvas inspection do not verify hands-on driving feel, audible output, or every spectator keyboard interaction. No human lap records were written by AI evaluation/playback. Generated checkpoints, images, plots, logs, and virtual environments are intentionally ignored by Git.
