from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
import torch
from stable_baselines3 import PPO
from apexrl.rl.bootstrap import bootstrap, fit_actor, teacher_action
from apexrl.rl.checkpoints import read_checkpoint
from apexrl.rl.config import load_training_config
from apexrl.rl.environment import ApexRacingEnv

ROOT = Path(__file__).resolve().parents[1]


def test_teacher_actions_are_bounded_and_seeded(tracks):
    env = ApexRacingEnv("velocity_circuit", tracks=tracks)
    try:
        env.reset(seed=126)
        action = teacher_action(env.simulation)
        assert action.dtype == np.float32 and env.action_space.contains(action)
        assert np.array_equal(action, teacher_action(env.simulation))
        with pytest.raises(ValueError, match="positive"):
            teacher_action(env.simulation, float("nan"))
    finally:
        env.close()


def test_actor_fit_learns_actions_without_changing_critic(tracks):
    torch.set_num_threads(1)
    env = ApexRacingEnv(tracks=tracks)
    try:
        model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, seed=126,
                    policy_kwargs={"net_arch": {"pi": [16,16], "vf": [16,16]}})
        rng = np.random.default_rng(126)
        observations = rng.uniform(-1, 1, (128,50)).astype(np.float32)
        targets = np.column_stack((observations[:,0]*.3, observations[:,1]*.5)).astype(np.float32)
        before = np.mean((model.predict(observations, deterministic=True)[0]-targets)**2)
        critic = {name: value.clone() for name, value in model.policy.state_dict().items()
                  if "value" in name}
        fit_actor(model, observations, targets, rng, updates=500)
        after = np.mean((model.predict(observations, deterministic=True)[0]-targets)**2)
        assert after < before*.2
        assert all(torch.equal(value, model.policy.state_dict()[name]) for name, value in critic.items())
    finally:
        env.close()


def test_bootstrap_reports_demonstrations_separately_from_ppo_steps(tmp_path, tracks):
    config = load_training_config(ROOT/"config/training-all.json")
    config["environment"].update(max_episode_seconds=.1, stall_seconds=.05)
    config["evaluation_episodes"] = 1
    config["ppo"].update(n_steps=16, batch_size=16, policy_width=16)
    result = bootstrap(root=ROOT, config=config, run_dir=tmp_path/"bootstrap", rounds=1,
                       episodes_per_track=1, updates=2)
    assert set(result["summary"]) == {"rookie_ring", "apex_valley", "velocity_circuit"}
    _, metadata, _ = read_checkpoint(tmp_path/"bootstrap/best_model.zip", ROOT, tracks)
    assert metadata["num_timesteps"] == 0
    assert metadata["bootstrap"]["demonstration_transitions"] > 0
    with pytest.raises(ValueError, match="not empty"):
        bootstrap(root=ROOT, config=config, run_dir=tmp_path/"bootstrap")
