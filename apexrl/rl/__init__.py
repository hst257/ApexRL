"""Optional Phase 2 modules. Importing apexrl itself does not import this package."""

from .environment import ApexRacingEnv
import gymnasium as gym

if "ApexRL-v0" not in gym.registry:
    gym.register("ApexRL-v0", entry_point="apexrl.rl.environment:ApexRacingEnv")

__all__ = ["ApexRacingEnv"]
