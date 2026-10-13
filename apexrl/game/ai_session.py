"""Lazy policy playback using the unchanged Gymnasium episode contract."""
from pathlib import Path
from apexrl.core import RacingSimulation


class AISession:
    def __init__(self, root: Path, model_path: Path, track_id: str, tracks=None, seed: int = 42):
        import torch
        from stable_baselines3 import PPO
        from apexrl.rl.checkpoints import read_checkpoint
        from apexrl.rl.environment import ApexRacingEnv

        torch.set_num_threads(1)
        checked, self.metadata, config = read_checkpoint(model_path, root, tracks)
        self.model = PPO.load(checked, device="cpu")
        self.env = ApexRacingEnv(track_id, root=root, config=config, tracks=tracks)
        if self.model.observation_space != self.env.observation_space or self.model.action_space != self.env.action_space:
            self.env.close()
            raise ValueError("Saved model spaces do not match the declared environment contract")
        self.seed = seed
        self.step_seconds = config.action_repeat/120
        self.reset()

    @property
    def simulation(self) -> RacingSimulation:
        return self.env.simulation

    def reset(self) -> None:
        self.observation, self.info = self.env.reset(seed=self.seed, options={"evaluation": True})
        self.finished = False

    def step(self) -> dict:
        if not self.finished:
            action, _ = self.model.predict(self.observation, deterministic=True)
            self.observation, _, terminated, truncated, self.info = self.env.step(action)
            self.finished = terminated or truncated
        return self.info

    def close(self) -> None:
        self.env.close()
