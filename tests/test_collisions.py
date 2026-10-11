from dataclasses import replace
import math
import pytest
from apexrl.core import RacingSimulation, VehicleControl
from apexrl.core.collisions import resolve_barriers
from apexrl.core.simulation import SimulationConfig
from apexrl.core.vehicle import VehicleState


def test_corner_footprint_hits_barrier_before_car_center(tracks, vehicle_config):
    track = tracks[0]
    x, y, angle = track.pose(60)
    nx, ny = -math.sin(angle), math.cos(angle)
    width = track.project(x, y).width
    offset = width/2+track.RUNOFF_WIDTH-.5
    state = VehicleState(x=x+nx*offset, y=y+ny*offset, heading=angle, lateral_velocity=20)
    previous = replace(state, x=state.x-nx*.2, y=state.y-ny*.2)
    assert track.project(state.x, state.y).distance < width/2+track.RUNOFF_WIDTH
    assert resolve_barriers(state, previous, track, vehicle_config)
    assert state.speed < 20


def test_sweep_prevents_barrier_tunneling(tracks, vehicle_config):
    track = tracks[0]
    x, y, angle = track.pose(70)
    nx, ny = -math.sin(angle), math.cos(angle)
    previous = VehicleState(x=x, y=y, heading=angle)
    state = VehicleState(x=x+nx*40, y=y+ny*40, heading=angle, lateral_velocity=100)
    assert resolve_barriers(state, previous, track, vehicle_config)
    assert track.project(state.x, state.y).distance < track.project(x,y).width/2+track.RUNOFF_WIDTH


def test_manual_recovery_invalidates_and_preserves_clock(tracks):
    simulation = RacingSimulation(tracks[0], simulation_config=SimulationConfig(countdown_seconds=0))
    for _ in range(100):
        simulation.step(VehicleControl(throttle=1))
    before = simulation.snapshot()
    result = simulation.recover_car()
    assert not result.state.lap_valid
    assert result.state.lap_time == before.lap_time
    assert result.state.vehicle.speed == 0
    assert all(tracks[0].surface(*point) == "asphalt" for point in simulation.vehicle.footprint(simulation.config))


def test_extended_grass_invalidates_then_recovers(tracks):
    simulation = RacingSimulation(tracks[0], simulation_config=SimulationConfig(countdown_seconds=0, shortcut_seconds=.1, recovery_seconds=.3))
    x, y, angle = tracks[0].pose(70)
    simulation.vehicle = VehicleState(x=x-math.sin(angle)*13, y=y+math.cos(angle)*13, heading=angle)
    simulation.projection = tracks[0].project(simulation.vehicle.x, simulation.vehicle.y)
    simulation.wheel_surfaces = ("grass",)*4
    events = []
    for _ in range(40):
        events.extend(simulation.step(VehicleControl()).events)
    assert not simulation.timing.valid
    assert any(event.kind == "recovered" for event in events)
    assert simulation.snapshot().surface == "asphalt"
