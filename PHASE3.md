# Phase 3: Driver Training and AI Garage

Phase 3 trains the existing PPO driver for longer, measures actual valid laps, and brings policy playback into the game. It preserves the Phase 2 observation/action contract and Phase 1 physics. RL dependencies remain optional for manual play. The published local Rookie Ring driver passed twenty of twenty held-out laps and the 90% publication gate. The harder circuits remain unproven.

## AI Garage

Launch the game using the RL virtual environment and choose **AI Garage** from the main menu:

```powershell
.\.venv\Scripts\python.exe main.py --ai-garage
```

The garage discovers local `.zip` checkpoints with `.metadata.json` sidecars in `runs/` and `models/`. It shows their training steps, inherited source steps, training circuit, and publication validation for the selected circuit. Use Next/Previous Policy, Change Circuit, Refresh Models, and Watch Policy. Published models in `models/` appear first. Discovery reads metadata without importing neural-network packages; compatibility and checksum checks run before model loading.

Watch only trusted checkpoints. If optional dependencies are missing, the garage shows installation instructions while manual gameplay remains available. Missing, corrupt, or incompatible models produce a visible error instead of starting playback.

```powershell
.\.venv\Scripts\python.exe main.py --ai-model models/rookie-reliable-v1.zip --track rookie_ring
```

The model receives the same observations and advances the same Gymnasium episode used for evaluation. **Escape** pauses/resumes; **R** restarts the episode; **Tab** expands the circuit map; **F3** toggles telemetry. Loss of focus pauses playback. A terminal result stays visible until restart or exit. The pause menu returns to the garage or main menu. AI session best times are displayed separately; AI laps never update human personal bests.

## Longer training

```powershell
.\.venv\Scripts\python.exe train.py --track rookie_ring --timesteps 200000 --run-dir runs/phase3-rookie-baseline
.\.venv\Scripts\python.exe train.py --track rookie_ring --timesteps 300000 --resume runs/phase3-rookie-baseline/best_model.zip --run-dir runs/phase3-rookie-refine
.\.venv\Scripts\python.exe plot_training.py runs/phase3-rookie-refine
```

Keep run directories separate. Resume restores the saved best policy and optimizer state; its starting step count can precede the first run's final count. PPO rounds requested transitions to full rollouts. Reports distinguish sampled training successes from deterministic evaluation success. Seeing a successful training episode does not demonstrate a reliable saved driver.

## Reliability training

The publication check exposed failures around the same Rookie Ring corner. `config/training-reliable.json` transfers the lap-completing policy into a fresh optimizer with a lower learning rate, four environments, a fresh seed, and initial action standard deviation of 0.05. It keeps the original reward and physics.

```powershell
.\.venv\Scripts\python.exe train.py --config config/training-reliable.json --initialize-from runs/phase3-rookie-refine/best_model.zip --run-dir runs/phase3-rookie-reliable
```

Use periodic evaluations to select a candidate; then run publication on fresh seeds that have not selected that checkpoint. A single successful canonical start is insufficient evidence of reliability.

## Corner-speed experiment

The original baseline reached roughly 44% of Rookie Ring before a collision. `config/training-corners.json` adds a curvature-based overspeed cost to encourage braking before tight corners. It samples upcoming curvature within a speed-dependent horizon, computes a conservative speed from an 8 m/s² lateral-acceleration budget, and penalizes squared excess speed. The original config keeps this penalty at zero; vehicle physics, observations, and actions are unchanged.

```powershell
.\.venv\Scripts\python.exe train.py --config config/training-corners.json --initialize-from runs/phase3-rookie-baseline/best_model.zip --run-dir runs/phase3-rookie-corners
```

`--initialize-from` copies policy weights into a fresh optimizer and resets this run's transition count. It permits reward tuning while requiring matching observation scales, episode settings, and architecture. The source and its training count are recorded. The corner config lowers learning rate and entropy incentive and sets initial action standard deviation to 0.2 after transfer. These are measured experiments, not a claim that this setting is optimal. Ordinary `--resume` still requires matching rewards and restores optimizer state. The two options are mutually exclusive.

## Publish a tested local driver

`promote.py` evaluates a checkpoint on seeds distinct from the training/scheduled evaluation seeds before placing it in the local garage's `models/` directory:

```powershell
.\.venv\Scripts\python.exe promote.py --model runs/phase3-rookie-reliable/checkpoints/ppo_50000.zip --name rookie-reliable-v1 --track rookie_ring --episodes 20 --seed 400084 --minimum-success .9
```

Default publication requires valid laps in at least 90% of ten seeded episodes on every selected circuit. At least five episodes are required. A failed gate saves a report and exits with status 2 without publishing. A passing gate copies the model, compatible sidecar, and measured evaluation report under a new name; existing published names are protected from overwrites. The sidecar records the source, evaluation seed, threshold, and per-circuit results.

The example has already produced `models/rookie-reliable-v1.zip` locally; choose a new model name for another publication. Model binaries are ignored by Git, so a fresh clone must execute the training stages to create its own driver.

Publication is local. Model files remain ignored by Git. Approval on Rookie Ring does not establish competence on Apex Valley or Velocity Circuit; evaluate those separately or use `--track all`. Ten episodes are a bounded check, not a statistical guarantee. See [PHASE3_VALIDATION.md](PHASE3_VALIDATION.md) for actual training and verification results.
