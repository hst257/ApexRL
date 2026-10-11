# Phase 2: Reinforcement Learning

Phase 2 adds a Gymnasium environment and Stable-Baselines3 PPO pipeline to the existing game. Manual gameplay retains its original controls, physics, and records. Training runs headlessly; rendering is optional. The supplied smoke checkpoint demonstrates the pipeline and still needs longer training to complete laps.

## Install optional dependencies

Use the repository root and Python 3.11+ on Windows. These commands choose the smaller CPU PyTorch distribution; CPU is the default training device for the small policy network.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
.\.venv\Scripts\python.exe -m pip check
```

The base `requirements.txt` remains sufficient for the standalone game. RL tests skip when their optional dependencies are absent. CUDA requires a compatible PyTorch distribution before selecting `--device cuda`.

## Train and inspect

```powershell
.\.venv\Scripts\python.exe train.py --smoke --run-dir runs/my-smoke
.\.venv\Scripts\python.exe train.py --track rookie_ring --timesteps 200000 --run-dir runs/rookie-200k
.\.venv\Scripts\python.exe evaluate.py --model runs/rookie-200k/final_model.zip --track all --episodes 10 --output runs/rookie-200k/evaluation.json
.\.venv\Scripts\python.exe watch.py --model runs/rookie-200k/best_model.zip --track rookie_ring
.\.venv\Scripts\python.exe plot_training.py runs/rookie-200k
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs --host 127.0.0.1 --port 6006
```

Open TensorBoard at `http://127.0.0.1:6006`. The viewer uses Escape to exit, Tab for the circuit map, and F3 for telemetry. `--frames 120 --screenshot artifacts/ai.png` bounds playback and captures its final canvas. Viewer steps follow the policy rather than keyboard driving and never update human lap records.

`config/training.json` controls rewards, episode rules, lookahead distances, PPO settings, seeds, checkpoint intervals, and evaluation intervals. Defaults train Rookie Ring for 200,000 requested transitions with two environments and a 64×64 actor/critic MLP. `--smoke` requests 16,384 transitions. PPO completes whole rollouts, so actual counts round up to multiples of `n_steps × n_envs`.

`--track all` samples among the three circuits at reset. `--vector subproc` runs independent Windows spawn workers; the default `dummy` vector batches two environments in one process, avoiding process overhead. Use a fresh run directory: existing results are protected from overwrites. Training steps count transitions across all environments, rather than physics ticks.

## Environment contract

```python
import gymnasium as gym
import apexrl.rl  # registers ApexRL-v0

env = gym.make("ApexRL-v0", track_id="rookie_ring")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step([0.1, 0.8])
env.close()
```

Actions are two continuous values in `[-1, 1]`: steering and signed pedal. Positive pedal means throttle; negative means brake. This maps onto the existing three-field `VehicleControl` without simultaneous acceleration and braking. Each action advances four 1/120-second physics ticks by default, yielding 30 decisions per simulated second. Terminal events stop the action early.

The default observation has 50 bounded float32 values: vehicle motion and controls, heading relative to the road, lateral offset, wheel surfaces, next-checkpoint distance, progress, remaining episode budgets, and six road lookaheads. Each lookahead includes local geometry, heading sine/cosine, and curvature. Normalization uses fixed physical scales; it has no learned running statistics. Names and contract versions live in `apexrl/rl/observations.py`.

Reward combines signed progress (0.1/m), time cost (0.05/s), grass/slip costs, steering-change cost, checkpoint bonuses (2), valid-lap bonus (100), and failure cost (10). Lap rules use the original ordered gates and forward finish crossing. Invalid progress and recovery teleports receive no progress reward.

A valid lap terminates successfully. Collisions, invalid laps, one second fully off-road, two seconds driving backward, or eight seconds without new forward progress terminate with failure. The 180-second horizon truncates the episode. Call `reset()` after either outcome. Seeded grid jitter varies training starts; `reset(options={"evaluation": True})` uses the canonical grid pose. Default CLI evaluation uses seeded jitter; `--canonical-start` disables it.

## Results and checkpoints

Each run stores `run.json`, `episodes.csv`, Monitor logs, TensorBoard events, periodic checkpoints, `best_model.zip`, `final_model.zip`, baseline/final evaluation JSON, and evaluation trajectories. Best-model selection compares deterministic mean return on the training circuits at scheduled evaluations and after the final update. Evaluation reports success, actual valid-lap times, progress, returns, and failure reasons separately; reward alone is insufficient evidence of driving skill.

Every model has a `.metadata.json` sidecar containing a checksum, configuration, seeds, dependency versions, track versions, physics fingerprint, and observation/action versions. Keep the pair together. Evaluation and playback reject incompatible contracts before loading. Bump contract versions when action or observation semantics change. Only load trusted model files; the sidecar establishes compatibility, not provenance.

Resume with matching environment, training track, and PPO hyperparameters:

```powershell
.\.venv\Scripts\python.exe train.py --smoke --timesteps 1024 --resume runs/my-smoke/final_model.zip --run-dir runs/my-resume
```

Resume restores policy and optimizer state and adds the requested transitions to the saved total. Environments and random streams restart, so resumed execution is not identical to one uninterrupted run. Ctrl+C saves the current policy, evaluates it, and closes environments.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe rl_benchmark.py --steps 10000 --output artifacts/rl-benchmark.json
.\.venv\Scripts\python.exe main.py
```

The RL benchmark includes observations, rewards, physics, and episode resets on all three circuits; it excludes neural inference and rendering. See [PHASE2_VALIDATION.md](PHASE2_VALIDATION.md) for measured results and current policy limitations.
