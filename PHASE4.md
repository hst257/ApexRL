# Phase 4: Drivers for Every Circuit

Phase 4 extends driver training to Rookie Ring, Apex Valley, and Velocity Circuit. The published local `all-circuits-v1` driver completed 20/20 held-out laps on each circuit. Manual gameplay, vehicle physics, and observation/action versions remain unchanged. Neural-network dependencies remain optional.

## Train across circuits

```powershell
.\.venv\Scripts\python.exe bootstrap.py --initialize-from models/rookie-reliable-v1.zip --run-dir runs/phase4-bootstrap
.\.venv\Scripts\python.exe train.py --config config/training-all-reliable.json --track all --initialize-from runs/phase4-bootstrap/checkpoints/actor_round_1.zip --run-dir runs/phase4-all-refine
.\.venv\Scripts\python.exe plot_training.py runs/phase4-all-refine
```

Each PPO environment selects a circuit with its seeded generator when resetting. Refinement uses four environments and a fresh optimizer. Choose new run directories for repeated experiments. See Phase 3 for creating the source policy in a fresh clone; model binaries are ignored by Git.

## Select a balanced checkpoint

Checkpoint selection compares, in order: the lowest circuit success rate, average success rate, lowest circuit progress, average progress, and average reward. A high reward on a long circuit cannot compensate for failed laps on another circuit. The starting policy is evaluated on the same scheduled seeds and retained if training never improves its ranking. Scheduled evaluation selects candidates; it does not provide independent publication evidence.

## Bootstrap and refine

The conservative teacher uses pure pursuit and upcoming curvature to generate demonstrations. Later rounds label states visited by a mixture of teacher and learner actions (DAgger). Only the actor is fitted; PPO refinement starts with a fresh optimizer. Checkpoints record demonstration transitions and PPO transitions separately. The teacher has privileged access to circuit geometry during training; game playback runs the learned actor using the unchanged 50 observations and two actions, without a teacher fallback.

## Validate and publish

```powershell
.\.venv\Scripts\python.exe promote.py --model runs/phase4-all-refine/checkpoints/ppo_30000.zip --name all-circuits-v1 --track all --episodes 20 --seed 900168 --minimum-success .9
.\.venv\Scripts\python.exe evaluate.py --model models/all-circuits-v1.zip --track all --episodes 5 --seed 1100168 --output artifacts/phase4-confirmation.json
```

Publication requires at least 90% valid laps on **each** selected circuit. Failure saves a report without publishing. The example freezes a scheduled checkpoint before checking fresh seeds. The local name already exists; choose a new name for another publication. Keep confirmation seeds separate from selection and publication seeds.

## AI Garage

```powershell
.\.venv\Scripts\python.exe main.py --ai-garage
```

Published drivers appear first, ordered by the number of circuits with at least 90% recorded validation success, then training steps. Change Circuit to review its measured validation result and watch the selected driver. Rookie specialists remain available. AI laps do not overwrite human lap records.

## Unsuccessful baseline

`config/training-all.json` also supports direct PPO transfer from the Rookie policy using `train.py --track all --initialize-from models/rookie-reliable-v1.zip`. The measured 401,408-step run failed the harder circuits and regressed on Rookie Ring. Use the demonstration-backed workflow above to reproduce the published driver.

See `PHASE4_VALIDATION.md` for measured results and any remaining limitations.
