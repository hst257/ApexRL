"""Train a policy from a geometric teacher; playback uses only learned observations."""
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from apexrl.core import load_tracks
from .checkpoints import make_manifest, read_checkpoint, save_checkpoint, write_json
from .config import EnvConfig
from .environment import ApexRacingEnv, TRACK_IDS
from .evaluation import evaluate
from .training import checkpoint_score


def teacher_action(simulation, cruise_speed: float = 28.) -> np.ndarray:
    """Pure pursuit with curvature-limited braking; privileged training teacher."""
    vehicle, track, projection = simulation.vehicle, simulation.track, simulation.projection
    config = simulation.config
    if not math.isfinite(cruise_speed) or cruise_speed <= 0 or config.maximum_steering_angle <= 0:
        raise ValueError("Teacher requires positive cruise speed and steering angle")
    speed = max(0., vehicle.longitudinal_velocity)
    x, y, _ = track.pose(projection.progress+max(8., speed*.6))
    dx, dy = x-vehicle.x, y-vehicle.y
    lateral = -dx*math.sin(vehicle.heading)+dy*math.cos(vehicle.heading)
    curvature = 2*lateral/max(dx*dx+dy*dy, 1.)
    steering = math.atan(config.wheelbase*curvature)/config.maximum_steering_angle
    distances = np.linspace(0, max(40., speed*3), 40)
    indices = np.searchsorted(track.cumulative, (projection.progress+distances)%track.total_length, side="right")-1
    limits = np.sqrt(5./np.maximum(np.abs(track.curvatures[indices]), .001))
    limits = np.sqrt(limits**2+12*np.maximum(0, distances-8))
    target = min(cruise_speed, float(limits.min()))
    return np.clip([steering, (target-speed)*.25], -1, 1).astype(np.float32)


def fit_actor(model, observations, actions, rng, updates: int = 2500) -> float:
    """Fit only the actor; the PPO value function and optimizer are untouched."""
    optimizer = torch.optim.Adam([*model.policy.mlp_extractor.policy_net.parameters(),
                                  *model.policy.action_net.parameters()], lr=.0003)
    model.policy.set_training_mode(True)
    inputs = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    targets = torch.as_tensor(actions, dtype=torch.float32, device=model.device)
    weights = torch.tensor([10., 1.], device=model.device)
    for _ in range(updates):
        indices = torch.as_tensor(rng.integers(len(inputs), size=256), device=model.device)
        latent = model.policy.mlp_extractor.forward_actor(model.policy.extract_features(inputs[indices]))
        predicted = model.policy.action_net(latent)
        loss = ((predicted-targets[indices]).square()*weights).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.)
        optimizer.step()
    model.policy.set_training_mode(False)
    return float(loss.detach().cpu())


def bootstrap(*, root: Path, config: dict, run_dir: Path, source: Path | None = None,
              rounds: int = 5, episodes_per_track: int = 3, updates: int = 2500) -> dict:
    if min(rounds, episodes_per_track, updates) < 1:
        raise ValueError("Rounds, episodes, and updates must be positive")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Run directory is not empty")
    env_config = EnvConfig.from_dict(config["environment"])
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config["seed"])
    tracks = load_tracks(root/"tracks")
    env = ApexRacingEnv("all", root=root, config=env_config, tracks=tracks)
    manifest = make_manifest(root, config, "all", tracks)
    try:
        parameters = dict(config["ppo"])
        width = parameters.pop("policy_width")
        model = PPO("MlpPolicy", env, seed=config["seed"], device="cpu",
                    policy_kwargs={"net_arch": {"pi": [width, width], "vf": [width, width]}}, **parameters)
        if source:
            checked, prior, prior_env = read_checkpoint(source, root, tracks)
            if prior_env != env_config or prior["training_config"]["ppo"]["policy_width"] != width:
                raise ValueError("Bootstrap source requires matching environment and architecture")
            previous = PPO.load(checked, device="cpu")
            model.policy.load_state_dict(previous.policy.state_dict())
            manifest.update(initialized_from=str(checked.resolve()), source_timesteps=prior["num_timesteps"])
            del previous
        with torch.no_grad():
            model.policy.log_std.fill_(math.log(config.get("initial_action_std", .05)))
        run_dir.mkdir(parents=True, exist_ok=True)
        observations, actions, results = [], [], []
        best = (-math.inf,)*5
        for iteration in range(rounds):
            # Teacher roll-ins first; later collect corrections on the learner's states (DAgger).
            learner_probability = 0. if iteration == 0 else min(.8, iteration*.2)
            print(f"Collecting bootstrap round {iteration+1}, learner probability {learner_probability:.0%}", flush=True)
            for track_id in TRACK_IDS:
                for episode in range(episodes_per_track):
                    seed = config["seed"]+iteration*1000+TRACK_IDS.index(track_id)*10000+episode
                    observation, _ = env.reset(seed=seed, options={"track_id": track_id})
                    cruise = float(rng.uniform(24., 30.))
                    done = False
                    while not done:
                        target = teacher_action(env.simulation, cruise)
                        observations.append(observation.copy())
                        actions.append(target)
                        action = model.predict(observation, deterministic=True)[0] if rng.random() < learner_probability else target
                        observation, _, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
            loss = fit_actor(model, np.asarray(observations), np.asarray(actions), rng, updates)
            manifest["bootstrap"] = {"method": "pure_pursuit_dagger", "rounds": iteration+1,
                                     "demonstration_transitions": len(observations), "actor_updates": (iteration+1)*updates,
                                     "seed": config["seed"], "episodes_per_track": episodes_per_track}
            save_checkpoint(model, run_dir/f"checkpoints/actor_round_{iteration+1}", manifest)
            result = evaluate(model, root=root, config=env_config, track_id="all", episodes=config["evaluation_episodes"],
                              seed=config["seed"]+100000, tracks=tracks, collect_trajectories=True)
            result.update(round=iteration+1, actor_loss=loss, demonstration_transitions=len(observations))
            results.append(result)
            write_json(run_dir/"evaluations.json", {"evaluations": results})
            rank = checkpoint_score(result["summary"])
            if rank > best:
                best = rank
                save_checkpoint(model, run_dir/"best_model", manifest)
            print(f"Bootstrap round {iteration+1}: {len(observations)} demonstrations, "
                  f"loss {loss:.6f}, success { {k:v['success_rate'] for k,v in result['summary'].items()} }", flush=True)
        save_checkpoint(model, run_dir/"final_model", manifest)
        write_json(run_dir/"run.json", manifest)
        write_json(run_dir/"final_evaluation.json", results[-1])
        return results[-1]
    finally:
        env.close()
