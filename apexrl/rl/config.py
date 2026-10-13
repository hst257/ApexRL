from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RewardConfig:
    progress_per_meter: float = 0.1
    time_per_second: float = 0.05
    grass_per_second: float = 0.5
    slip_per_second: float = 0.15
    steering_change: float = 0.01
    checkpoint: float = 2.0
    lap: float = 100.0
    failure: float = 10.0
    corner_overspeed: float = 0.0
    corner_lateral_acceleration: float = 8.0

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Reward parameter {item.name} must be finite and nonnegative")
        if self.corner_lateral_acceleration <= 0:
            raise ValueError("corner_lateral_acceleration must be positive")


@dataclass(frozen=True, slots=True)
class EnvConfig:
    action_repeat: int = 4
    max_episode_seconds: float = 180.0
    offtrack_seconds: float = 1.0
    stall_seconds: float = 8.0
    wrong_way_seconds: float = 2.0
    progress_threshold: float = 0.5
    spawn_lateral_jitter: float = 0.3
    spawn_heading_jitter: float = 0.03
    lookahead_distances: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0, 100.0, 160.0)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        if not isinstance(self.action_repeat, int) or not 1 <= self.action_repeat <= 120:
            raise ValueError("action_repeat must be an integer in [1, 120]")
        for name in ("max_episode_seconds", "offtrack_seconds", "stall_seconds", "wrong_way_seconds", "progress_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, upper in (("spawn_lateral_jitter", 1.0), ("spawn_heading_jitter", .2)):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= upper:
                raise ValueError(f"Invalid {name}")
        distances = tuple(float(value) for value in self.lookahead_distances)
        if not distances or any(not math.isfinite(value) or value <= 0 for value in distances) or tuple(sorted(set(distances))) != distances:
            raise ValueError("Lookahead distances must be finite, positive, and strictly increasing")
        object.__setattr__(self, "lookahead_distances", distances)

    @classmethod
    def from_dict(cls, values: dict) -> "EnvConfig":
        values = dict(values)
        if "reward" in values:
            values["reward"] = RewardConfig(**values["reward"])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


def load_training_config(path: str | Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    EnvConfig.from_dict(result["environment"])
    if "initial_action_std" in result and (not math.isfinite(result["initial_action_std"]) or result["initial_action_std"] <= 0):
        raise ValueError("initial_action_std must be finite and positive")
    for name in ("total_timesteps", "n_envs", "checkpoint_frequency", "evaluation_frequency", "evaluation_episodes", "torch_threads"):
        if not isinstance(result[name], int) or result[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(result["seed"], int) or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    ppo = result["ppo"]
    for name in ("n_steps", "batch_size", "n_epochs", "policy_width"):
        if not isinstance(ppo[name], int) or ppo[name] <= 0:
            raise ValueError(f"PPO {name} must be a positive integer")
    if ppo["batch_size"] < 2 or ppo["n_steps"] < 2:
        raise ValueError("PPO batch size and rollout length must be at least two")
    for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"):
        if not math.isfinite(ppo[name]) or ppo[name] < 0:
            raise ValueError(f"Invalid PPO {name}")
    if not 0 < ppo["gamma"] <= 1 or not 0 < ppo["gae_lambda"] <= 1 or ppo["learning_rate"] <= 0:
        raise ValueError("Invalid learning rate or discount factors")
    return result
