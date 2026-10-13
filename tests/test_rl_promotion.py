import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from apexrl.rl import ApexRacingEnv
from apexrl.rl.checkpoints import make_manifest, save_checkpoint
from apexrl.rl.config import load_training_config
from apexrl.rl.promotion import publish_policy, qualifies
from apexrl.game.ai_session import AISession

ROOT = Path(__file__).resolve().parents[1]


def report(success=1.0):
    return {"seed": 200042, "summary": {"rookie_ring": {"success_rate": success, "best_lap_seconds": 60.0 if success else None}}}


def test_promotion_requires_measured_valid_laps_on_every_selected_track():
    assert qualifies(report(), .9)
    assert not qualifies(report(.8), .9)
    missing = report()
    missing["summary"]["rookie_ring"]["best_lap_seconds"] = None
    assert not qualifies(missing, .9)
    with pytest.raises(ValueError):
        qualifies(report(), 0)


def test_publication_keeps_pair_and_refuses_overwrite_or_wrong_report(tmp_path, monkeypatch):
    source = tmp_path/"run"/"best_model.zip"
    source.parent.mkdir()
    source.write_bytes(b"trained model")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr("apexrl.rl.promotion.read_checkpoint", lambda path, root: (source, {"model_sha256":digest}, None))
    evaluation = dict(report(), model_sha256=digest)
    published = publish_policy(source, tmp_path, "rookie-v1", evaluation, .9)
    assert published.read_bytes() == source.read_bytes()
    assert json.loads(published.with_suffix(".metadata.json").read_text())["promotion"]["evaluation_seed"] == 200042
    assert published.with_suffix(".evaluation.json").is_file()
    with pytest.raises(ValueError, match="exists"):
        publish_policy(source, tmp_path, "rookie-v1", evaluation, .9)
    with pytest.raises(ValueError, match="match"):
        publish_policy(source, tmp_path, "rookie-v2", dict(evaluation, model_sha256="wrong"), .9)
    with pytest.raises(ValueError, match="name"):
        publish_policy(source, tmp_path, "../escape", evaluation, .9)


def test_real_ai_session_uses_exact_saved_policy_actions_and_stops_at_terminal(tmp_path, tracks):
    import torch
    torch.set_num_threads(1)
    config = load_training_config(ROOT/"config/training.json")
    config["environment"]["max_episode_seconds"] = .05
    from apexrl.rl.config import EnvConfig
    env = ApexRacingEnv(config=EnvConfig.from_dict(config["environment"]), tracks=tracks)
    try:
        model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, device="cpu", seed=42)
        path = save_checkpoint(model, tmp_path/"policy", make_manifest(ROOT, config, "rookie_ring", tracks))
        session = AISession(ROOT, path, "rookie_ring", tracks)
        try:
            expected = model.predict(session.observation, deterministic=True)[0]
            assert np.array_equal(expected, session.model.predict(session.observation, deterministic=True)[0])
            session.step()
            info = session.step()
            assert session.finished and info["termination_reason"] == "time_limit"
            before = session.simulation.snapshot()
            session.step()
            assert session.simulation.snapshot() == before
            session.reset()
            assert not session.finished
        finally:
            session.close()
    finally:
        env.close()
