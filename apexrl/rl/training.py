import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from apexrl.core import load_tracks
from .checkpoints import make_manifest, read_checkpoint, save_checkpoint, write_json
from .config import EnvConfig
from .environment import ApexRacingEnv
from .evaluation import evaluate


def checkpoint_score(summary: dict) -> tuple[float, ...]:
    """Prefer reliable laps on every circuit before progress or raw reward."""
    values = list(summary.values())
    if not values:
        raise ValueError("Checkpoint selection requires at least one circuit")
    success = [value["success_rate"] for value in values]
    progress = [value["mean_max_progress_fraction"] for value in values]
    return (min(success), float(np.mean(success)), min(progress), float(np.mean(progress)),
            float(np.mean([value["mean_return"] for value in values])))


class EnvFactory:
    """Picklable Windows spawn factory. Never captures Pygame or simulation state."""
    def __init__(self, root, track_id, config, monitor_path, tracks=None):
        self.root, self.track_id, self.config = root, track_id, config
        self.monitor_path, self.tracks = monitor_path, tracks

    def __call__(self):
        return Monitor(ApexRacingEnv(self.track_id, root=self.root, config=self.config, tracks=self.tracks),
                       filename=str(self.monitor_path), info_keywords=("is_success", "progress_fraction", "termination_reason"))


def make_vector_env(root, track_id, config, count, run_dir, seed, vector="dummy", tracks=None):
    if count < 1 or vector not in ("dummy", "subproc"):
        raise ValueError("Invalid vector environment configuration")
    factories = [EnvFactory(root, track_id, config, run_dir/f"monitor_{i}.csv", tracks if vector == "dummy" else None)
                 for i in range(count)]
    env = DummyVecEnv(factories) if vector == "dummy" else SubprocVecEnv(factories, start_method="spawn")
    env.seed(seed)
    return env


class RacingCallback(BaseCallback):
    def __init__(self, root, run_dir, config, manifest, tracks):
        super().__init__()
        self.root, self.run_dir, self.config = root, run_dir, config
        self.manifest, self.tracks = manifest, tracks
        self.best_score = (-math.inf,)*5

    def _on_training_start(self):
        self.next_checkpoint = self.num_timesteps+self.config["checkpoint_frequency"]
        self.next_evaluation = self.num_timesteps+self.config["evaluation_frequency"]
        self.episode_file = (self.run_dir/"episodes.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.episode_file, fieldnames=("timesteps", "track", "return", "length", "seconds", "max_progress", "success", "reason"))
        self.writer.writeheader()
        self.evaluations = []

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode" in info:
                self.writer.writerow({"timesteps": self.num_timesteps, "track": info["track_id"], "return": info["episode"]["r"],
                                      "length": info["episode"]["l"], "seconds": info["episode_seconds"],
                                      "max_progress": info.get("max_progress_fraction", info["progress_fraction"]),
                                      "success": int(info["is_success"]), "reason": info["termination_reason"]})
        if self.num_timesteps >= self.next_checkpoint:
            save_checkpoint(self.model, self.run_dir/f"checkpoints/ppo_{self.num_timesteps}", self.manifest)
            self.episode_file.flush()
            self.next_checkpoint += self.config["checkpoint_frequency"]
        if self.num_timesteps >= self.next_evaluation:
            result = evaluate(self.model, root=self.root, config=EnvConfig.from_dict(self.config["environment"]),
                              track_id=self.manifest["training_track"], episodes=self.config["evaluation_episodes"],
                              seed=self.config["seed"]+100000, tracks=self.tracks)
            result["timesteps"] = self.num_timesteps
            self.evaluations.append(result)
            write_json(self.run_dir/"evaluations.json", {"evaluations": self.evaluations})
            success = float(np.mean([value["success_rate"] for value in result["summary"].values()]))
            score = float(np.mean([value["mean_return"] for value in result["summary"].values()]))
            self.logger.record("eval/success_rate", success)
            self.logger.record("eval/mean_reward", score)
            rank = checkpoint_score(result["summary"])
            self.logger.record("eval/weakest_circuit_success", rank[0])
            for identifier, value in result["summary"].items():
                self.logger.record(f"eval/{identifier}/success_rate", value["success_rate"])
                self.logger.record(f"eval/{identifier}/max_progress", value["mean_max_progress_fraction"])
            if rank > self.best_score:
                self.best_score = rank
                save_checkpoint(self.model, self.run_dir/"best_model", self.manifest)
            print(f"Evaluation at {self.num_timesteps}: mean reward {score:.2f}, success {success:.0%}, weakest circuit {rank[0]:.0%}", flush=True)
            self.next_evaluation += self.config["evaluation_frequency"]
        return True

    def _on_training_end(self):
        self.episode_file.close()


def train(*, root, config, track_id="rookie_ring", run_dir=None, resume=None, vector="dummy", device="cpu", initialize_from=None):
    if resume and initialize_from:
        raise ValueError("Choose resume or initialize_from, not both")
    if "initial_action_std" in config and (not math.isfinite(config["initial_action_std"]) or config["initial_action_std"] <= 0):
        raise ValueError("initial_action_std must be finite and positive")
    root = Path(root)
    run_dir = Path(run_dir) if run_dir else root/"runs"/f"ppo_{track_id}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Run directory is not empty; choose a new directory to preserve previous results")
    env_config = EnvConfig.from_dict(config["environment"])
    rollout = config["ppo"]["n_steps"]*config["n_envs"]
    if config["ppo"]["batch_size"] > rollout or rollout % config["ppo"]["batch_size"]:
        raise ValueError("PPO batch_size must divide n_steps × n_envs")
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    tracks = load_tracks(root/"tracks")
    manifest = make_manifest(root, config, track_id, tracks)
    # Validate before creating run output or beginning optimization.
    checker = ApexRacingEnv(track_id, root=root, config=env_config, tracks=tracks)
    try:
        check_env(checker, warn=True, skip_render_check=True)
    finally:
        checker.close()
    if resume:
        resume_path, prior, prior_env = read_checkpoint(resume, root, tracks)
        if prior_env != env_config or prior["training_track"] != track_id or prior["training_config"]["ppo"] != config["ppo"]:
            raise ValueError("Resume requires matching environment, track selection, and PPO hyperparameters")
        manifest["resumed_from"] = str(resume_path.resolve())
    if initialize_from:
        initialize_path, prior, prior_env = read_checkpoint(initialize_from, root, tracks)
        expected, previous = env_config.to_dict(), prior_env.to_dict()
        expected.pop("reward")
        previous.pop("reward")
        if expected != previous or prior["training_config"]["ppo"]["policy_width"] != config["ppo"]["policy_width"]:
            raise ValueError("Initialization requires matching observation scales, episode settings, and policy architecture")
        manifest["initialized_from"] = str(initialize_path.resolve())
        manifest["source_timesteps"] = prior["num_timesteps"]
        if "bootstrap" in prior:
            manifest["source_bootstrap"] = prior["bootstrap"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir/"run.json", manifest)
    env = make_vector_env(root, track_id, env_config, config["n_envs"], run_dir, config["seed"], vector, tracks)
    callback = RacingCallback(root, run_dir, config, manifest, tracks)
    model = None
    interrupted = False
    start = time.perf_counter()
    try:
        if resume:
            model = PPO.load(resume_path, env=env, device=device, tensorboard_log=str(run_dir/"tensorboard"))
        else:
            parameters = dict(config["ppo"])
            width = parameters.pop("policy_width")
            model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": {"pi": [width, width], "vf": [width, width]}},
                        seed=config["seed"], device=device, verbose=1, tensorboard_log=str(run_dir/"tensorboard"), **parameters)
            if initialize_from:
                source = PPO.load(initialize_path, device=device)
                model.policy.load_state_dict(source.policy.state_dict())
                del source
                if "initial_action_std" in config:
                    std = config["initial_action_std"]
                    with torch.no_grad():
                        model.policy.log_std.fill_(math.log(std))
        initial_steps = model.num_timesteps
        baseline = evaluate(model, root=root, config=env_config, track_id="all", episodes=config["evaluation_episodes"], seed=config["seed"]+100000, tracks=tracks)
        write_json(run_dir/"baseline_evaluation.json", baseline)
        callback.best_score = checkpoint_score({key: value for key, value in baseline["summary"].items()
                                               if track_id == "all" or key == track_id})
        save_checkpoint(model, run_dir/"best_model", manifest)
        try:
            model.learn(total_timesteps=config["total_timesteps"], callback=callback,
                        reset_num_timesteps=not bool(resume), tb_log_name="ppo")
        except KeyboardInterrupt:
            interrupted = True
            print("Training interrupted; saving completed updates and the current policy.", flush=True)
        save_checkpoint(model, run_dir/"final_model", manifest)
        result = evaluate(model, root=root, config=env_config, track_id="all", episodes=config["evaluation_episodes"],
                          seed=config["seed"]+100000, tracks=tracks, collect_trajectories=True)
        selected = {key: value for key, value in result["summary"].items() if track_id == "all" or key == track_id}
        final_score = checkpoint_score(selected)
        if final_score > callback.best_score:
            callback.best_score = final_score
            save_checkpoint(model, run_dir/"best_model", manifest)
        elapsed = time.perf_counter()-start
        result.update({"num_timesteps": model.num_timesteps, "new_timesteps": model.num_timesteps-initial_steps,
                       "wall_seconds_including_evaluation": elapsed, "interrupted": interrupted})
        write_json(run_dir/"final_evaluation.json", result)
        print(f"Saved policy and evaluation to {run_dir.resolve()}", flush=True)
        return run_dir, result
    finally:
        if hasattr(callback, "episode_file") and not callback.episode_file.closed:
            callback.episode_file.close()
        env.close()
