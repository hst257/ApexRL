from pathlib import Path
import json
import numpy as np
import pytest

pytest.importorskip("stable_baselines3", reason="Install requirements-rl.txt for training tests")
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from apexrl.rl import ApexRacingEnv
from apexrl.rl.checkpoints import make_manifest, read_checkpoint, save_checkpoint
from apexrl.rl.config import EnvConfig, load_training_config
from apexrl.rl.evaluation import evaluate
from apexrl.rl.training import make_vector_env, train

ROOT = Path(__file__).resolve().parents[1]


def test_sb3_checker(tracks):
    check_env(ApexRacingEnv(tracks=tracks), warn=True, skip_render_check=True)


def test_short_ppo_update_reload_and_contract_rejection(tmp_path, tracks):
    torch.set_num_threads(1)
    config = load_training_config(ROOT/"config/training.json")
    config["environment"]["max_episode_seconds"] = .3
    config["environment"]["stall_seconds"] = .2
    env_config = EnvConfig.from_dict(config["environment"])
    vector = make_vector_env(ROOT, "rookie_ring", env_config, 2, tmp_path, 42, tracks=tracks)
    try:
        model = PPO("MlpPolicy", vector, n_steps=16, batch_size=16, n_epochs=1, seed=42, device="cpu", policy_kwargs={"net_arch":[16,16]})
        weights = [parameter.detach().clone() for parameter in model.policy.parameters()]
        model.learn(64)
        assert any(not torch.equal(before, after) for before, after in zip(weights, model.policy.parameters()))
        manifest = make_manifest(ROOT, config, "rookie_ring", tracks)
        path = save_checkpoint(model, tmp_path/"policy", manifest)
        checked, metadata, loaded_config = read_checkpoint(path, ROOT, tracks)
        assert metadata["num_timesteps"] == 64
        loaded = PPO.load(checked, device="cpu")
        obs = vector.reset()
        assert np.array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
        report = evaluate(loaded, root=ROOT, config=loaded_config, episodes=1, tracks=tracks)
        repeat = evaluate(loaded, root=ROOT, config=loaded_config, episodes=1, tracks=tracks)
        assert report == repeat
        assert set(report["summary"]) == {"rookie_ring","apex_valley","velocity_circuit"}
        sidecar = path.with_suffix(".metadata.json")
        data = json.loads(sidecar.read_text())
        data["observation_version"] = 999
        sidecar.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="contract"):
            read_checkpoint(path, ROOT, tracks)
    finally:
        vector.close()


def test_spawn_vector_environment_resets_and_steps(tmp_path):
    config = EnvConfig(max_episode_seconds=.1)
    vector = make_vector_env(ROOT, "rookie_ring", config, 2, tmp_path, 123, vector="subproc")
    try:
        observations = vector.reset()
        assert observations.shape == (2,50)
        observations, rewards, dones, infos = vector.step(np.array([[0,.8],[0,.8]], dtype=np.float32))
        assert rewards.shape == (2,)
        assert observations.shape == (2,50)
    finally:
        vector.close()


def test_training_outputs_resume_and_preserves_existing_run(tmp_path, tracks):
    config = load_training_config(ROOT/"config/training.json")
    config.update(total_timesteps=32, n_envs=1, evaluation_episodes=1,
                  checkpoint_frequency=1000, evaluation_frequency=1000)
    config["ppo"].update(n_steps=16, batch_size=16, n_epochs=1, policy_width=16)
    config["environment"].update(max_episode_seconds=.2, stall_seconds=.1)
    run_dir, report = train(root=ROOT, config=config, run_dir=tmp_path/"first")
    assert report["new_timesteps"] == report["num_timesteps"] == 32
    assert (run_dir/"best_model.zip").is_file()
    assert list((run_dir/"tensorboard").rglob("events.out.tfevents.*"))
    assert (run_dir/"episodes.csv").read_text().count("\n") > 1
    _, metadata, _ = read_checkpoint(run_dir/"final_model.zip", ROOT, tracks)
    assert metadata["num_timesteps"] == 32
    with pytest.raises(ValueError, match="not empty"):
        train(root=ROOT, config=config, run_dir=run_dir)
    resumed, report = train(root=ROOT, config=config, resume=run_dir/"final_model.zip", run_dir=tmp_path/"resumed")
    assert report["new_timesteps"] == 32
    assert report["num_timesteps"] == 64
    assert json.loads((resumed/"run.json").read_text())["resumed_from"] == str((run_dir/"final_model.zip").resolve())
