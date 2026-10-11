import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="Install requirements-rl.txt for Phase 2 tests")
from gymnasium.utils.env_checker import check_env
from apexrl.core.vehicle import VehicleState
from apexrl.rl import ApexRacingEnv
from apexrl.rl.config import EnvConfig, RewardConfig

ROOT = Path(__file__).resolve().parents[1]


def make_env(tracks, **config):
    return ApexRacingEnv(tracks=tracks, config=EnvConfig(**config))


def test_gymnasium_contract_and_registered_environment(tracks):
    env = make_env(tracks)
    check_env(env, skip_render_check=True)
    assert env.observation_space.shape == (50,)
    assert len(env.observation_names) == 50
    wrapped = gym.make("ApexRL-v0", tracks=tracks)
    observation, info = wrapped.reset(seed=42)
    assert wrapped.observation_space.contains(observation)
    assert info["track_id"] == "rookie_ring"
    wrapped.close()


def test_seeded_multi_track_resets_and_instances_are_independent(tracks):
    a = ApexRacingEnv("all", tracks=tracks)
    b = ApexRacingEnv("all", tracks=tracks)
    for seed in (42, 17, 42):
        first, ia = a.reset(seed=seed)
        second, ib = b.reset(seed=seed)
        assert np.array_equal(first, second)
        assert ia == ib
        for _ in range(10):
            xa = a.step(np.array((.03, .8)))
            xb = b.step(np.array((.03, .8)))
            assert np.array_equal(xa[0], xb[0])
            assert xa[1:] == xb[1:]
    before = b.simulation.snapshot()
    a.reset(seed=5)
    assert b.simulation.snapshot() == before


def test_observation_buffers_are_owned_and_finite(tracks):
    env = make_env(tracks)
    first, _ = env.reset(seed=42)
    stored = first.copy()
    second, _, _, _, _ = env.step((0,1))
    assert np.array_equal(first, stored)
    assert not np.shares_memory(first, second)
    assert np.isfinite(second).all()
    assert second.dtype == np.float32


@pytest.mark.parametrize("action", [(float("nan"),0), (0,float("inf")), [0], [0,1,0]])
def test_invalid_actions_do_not_advance_state(tracks, action):
    env = make_env(tracks)
    env.reset(seed=42)
    before = env.simulation.snapshot()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.simulation.snapshot() == before


def test_actions_clamp_and_signed_pedal_maps_to_vehicle_interface(tracks):
    env = make_env(tracks)
    env.reset(seed=42)
    _, _, _, _, info = env.step((5,2))
    vehicle = env.simulation.snapshot().vehicle
    assert vehicle.steering_input == vehicle.throttle_input == 1
    assert vehicle.brake_input == 0
    assert info["physics_ticks"] == 4
    env.step((0,-2))
    assert env.simulation.vehicle.brake_input == 1
    assert env.simulation.vehicle.throttle_input == 0


def test_time_limit_is_truncation_and_stall_is_termination(tracks):
    timeout = make_env(tracks, max_episode_seconds=.05)
    timeout.reset(seed=0)
    timeout.step((0,0))
    _, _, terminated, truncated, info = timeout.step((0,0))
    assert not terminated and truncated
    assert info["termination_reason"] == "time_limit"
    assert info["episode_seconds"] == pytest.approx(.05)
    assert info["physics_ticks"] == 2
    with pytest.raises(gym.error.ResetNeeded):
        timeout.step((0,0))
    stalled = make_env(tracks, stall_seconds=.05)
    stalled.reset(seed=0)
    stalled.step((0,0))
    _, reward, terminated, truncated, info = stalled.step((0,0))
    assert terminated and not truncated
    assert info["termination_reason"] == "stalled"
    assert reward < -10


def test_step_before_reset_is_rejected(tracks):
    with pytest.raises(gym.error.ResetNeeded):
        make_env(tracks).step((0,0))


def test_signed_progress_rewards_forward_and_penalizes_reverse(tracks):
    env = make_env(tracks)
    env.reset(seed=0, options={"evaluation": True})
    _, _, _, _, info = env.step((0,1))
    assert info["reward_terms"]["progress"] > 0
    env.reset(seed=0, options={"evaluation": True})
    env.simulation.vehicle.heading += math.pi
    env.simulation.vehicle.longitudinal_velocity = 10
    _, _, _, _, info = env.step((0,0))
    assert info["reward_terms"]["progress"] < 0


def test_wrong_way_terminates(tracks):
    env = make_env(tracks, wrong_way_seconds=.05)
    env.reset(seed=0, options={"evaluation": True})
    env.simulation.vehicle.heading += math.pi
    env.simulation.vehicle.longitudinal_velocity = 10
    env.step((0,0))
    _, _, terminated, truncated, info = env.step((0,0))
    assert terminated and not truncated
    assert info["termination_reason"] == "wrong_way"


def place_on_grass(env, distance):
    track = env.simulation.track
    x, y, angle = track.pose(70)
    env.simulation.vehicle = VehicleState(x=x-math.sin(angle)*distance, y=y+math.cos(angle)*distance, heading=angle)
    env.simulation.projection = track.project(env.simulation.vehicle.x, env.simulation.vehicle.y)
    env.simulation.wheel_surfaces = tuple(track.surface(*p) for p in env.simulation.vehicle.footprint(env.vehicle_config, wheels=True))


def test_offtrack_and_collision_terminate_without_recovery_credit(tracks):
    env = make_env(tracks, offtrack_seconds=.025)
    env.reset(seed=0)
    place_on_grass(env, 13)
    _, _, terminated, truncated, info = env.step((0,0))
    assert terminated and not truncated
    assert info["termination_reason"] == "offtrack"
    env.reset(seed=0)
    place_on_grass(env, 17.4)
    env.simulation.vehicle.lateral_velocity = 20
    _, _, terminated, _, info = env.step((0,0))
    assert terminated
    assert info["termination_reason"] == "collision"
    assert info["reward_terms"]["failure"] == -10


def test_recovery_teleports_never_pay_progress(tracks, monkeypatch):
    env = make_env(tracks)
    env.reset(seed=0)
    env.simulation.timing.last_checkpoint = 3
    monkeypatch.setattr(env.simulation, "step", lambda action, dt: env.simulation.recover_car())
    _, _, terminated, _, info = env.step((0,1))
    assert terminated
    assert info["reward_terms"]["progress"] == 0
    assert info["termination_reason"] == "invalid_lap"


def test_only_valid_completed_lap_pays_success_bonus(tracks):
    env = make_env(tracks)
    env.reset(seed=0, options={"evaluation": True})
    sim, track = env.simulation, tracks[0]
    x, y, heading = track.pose(track.total_length-.05)
    sim.vehicle = VehicleState(x=x,y=y,heading=heading,longitudinal_velocity=12)
    sim.projection = track.project(x,y)
    sim.timing.next_checkpoint = len(track.checkpoints)
    sim.timing.forward_distance = track.total_length-.05
    sim.timing.finish_armed = True
    sim.timing.elapsed = 40
    _, reward, terminated, truncated, info = env.step((0,0))
    assert terminated and not truncated and info["is_success"]
    assert info["reward_terms"]["lap"] == 100
    assert info["lap_time"] == pytest.approx(40.0042, abs=.001)
    assert reward > 99


def test_headless_rl_does_not_import_pygame_or_torch():
    script = """
import sys
sys.modules['pygame'] = None
sys.modules['torch'] = None
from apexrl.rl import ApexRacingEnv
env = ApexRacingEnv()
env.reset(seed=42)
for _ in range(100):
    env.step((0,.8))
assert 'pygame.display' not in sys.modules
env.close()
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_rgb_render_does_not_create_window(tracks, monkeypatch):
    import pygame
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.display.quit()
    env = ApexRacingEnv(tracks=tracks, render_mode="rgb_array")
    try:
        env.reset(seed=42)
        before = env.simulation.snapshot()
        frame = env.render()
        assert frame.shape == (720,1280,3) and frame.dtype == np.uint8
        assert not pygame.display.get_init()
        assert env.simulation.snapshot() == before
    finally:
        env.close()


@pytest.mark.parametrize("values", [{"action_repeat":0}, {"stall_seconds":float("nan")}, {"offtrack_seconds":0}, {"lookahead_distances":[30,5]}, {"spawn_heading_jitter":1}])
def test_invalid_environment_configs_fail(values):
    with pytest.raises(ValueError):
        EnvConfig(**values)


def test_negative_reward_weights_rejected():
    with pytest.raises(ValueError):
        RewardConfig(progress_per_meter=-1)
