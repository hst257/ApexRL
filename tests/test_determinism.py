from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import random
import subprocess
import sys
import pytest
from apexrl.core import RacingSimulation, VehicleControl
from apexrl.core.simulation import SimulationConfig


def test_identical_actions_and_interleaved_instances_match(tracks):
    config = SimulationConfig(countdown_seconds=0)
    a, b = (RacingSimulation(tracks[0], seed=42, simulation_config=config) for _ in range(2))
    unrelated = RacingSimulation(tracks[1], seed=7, simulation_config=config)
    rng = random.Random(90)
    for _ in range(700):
        action = VehicleControl(rng.uniform(-.5,.5), rng.random(), rng.random()*.2)
        first = a.step(action)
        unrelated.step(VehicleControl(throttle=1, steering=.8))
        second = b.step(action)
        assert first == second
    a.reset()
    assert a.vehicle.speed == 0
    assert b.vehicle.speed > 0


def test_core_import_and_stepping_work_with_pygame_blocked():
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
sys.modules['pygame'] = None
from apexrl.core import load_tracks, RacingSimulation, VehicleControl
s = RacingSimulation(load_tracks('tracks')[0])
for _ in range(400):
    s.step(VehicleControl(throttle=1))
assert 'pygame.display' not in sys.modules
assert s.snapshot().vehicle.speed > 0
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dt", [0, -1, math.nan, math.inf, .1])
def test_bad_timestep_rejected_without_mutation(tracks, dt):
    simulation = RacingSimulation(tracks[0])
    before = simulation.snapshot()
    with pytest.raises(ValueError):
        simulation.step(VehicleControl(), dt)
    assert simulation.snapshot() == before


def test_countdown_holds_car_and_restart_clears_lap(tracks):
    simulation = RacingSimulation(tracks[0])
    for _ in range(360):
        simulation.step(VehicleControl(throttle=1))
    assert simulation.snapshot().vehicle.speed == 0
    assert simulation.snapshot().countdown == 0
    assert simulation.snapshot().lap_time == 0
    simulation.step(VehicleControl(throttle=1))
    assert simulation.snapshot().lap_time > 0
    simulation.recover_car()
    simulation.reset()
    assert simulation.snapshot().lap_valid
    assert simulation.snapshot().lap_time == 0
    assert simulation.snapshot().countdown == 3


def test_result_snapshots_are_immutable(tracks):
    state = RacingSimulation(tracks[0]).snapshot()
    with pytest.raises(FrozenInstanceError):
        state.vehicle.x = 99


@pytest.mark.parametrize("values", [{"shortcut_seconds": 0}, {"countdown_seconds": -1}, {"recovery_seconds": 1}, {"shortcut_seconds": math.nan}])
def test_invalid_race_durations_rejected(values):
    with pytest.raises(ValueError):
        SimulationConfig(**values)


def test_seeded_reset_offsets_validate_before_mutation(tracks):
    simulation = RacingSimulation(tracks[0])
    before = simulation.snapshot()
    with pytest.raises(ValueError, match="footprint"):
        simulation.reset(seed=99, lateral_offset=100)
    assert simulation.snapshot() == before
    assert simulation.seed == 42
    a = simulation.reset(seed=99, lateral_offset=.2, heading_offset=.02)
    b = simulation.reset(seed=99, lateral_offset=.2, heading_offset=.02)
    assert a == b
    assert simulation.seed == 99
