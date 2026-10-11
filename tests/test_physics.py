import math
import pytest
from apexrl.core.controls import VehicleControl
from apexrl.core.physics import integrate_vehicle
from apexrl.core.vehicle import VehicleState, VehicleConfig


def run(state, action, seconds, config, grip=1):
    for _ in range(round(seconds*120)):
        integrate_vehicle(state, action, config, 1/120, grip=grip)
    return state


def test_stationary_car_accelerates(vehicle_config):
    state = run(VehicleState(), VehicleControl(throttle=1), 2, vehicle_config)
    assert state.longitudinal_velocity > 20
    assert state.x > 15


def test_braking_stops_without_selecting_reverse(vehicle_config):
    state = run(VehicleState(longitudinal_velocity=60), VehicleControl(brake=1), 6, vehicle_config)
    assert state.longitudinal_velocity == 0
    run(state, VehicleControl(brake=1), 3, vehicle_config)
    assert state.longitudinal_velocity == 0


def test_drag_preserves_momentum_but_reduces_speed(vehicle_config):
    state = run(VehicleState(longitudinal_velocity=50), VehicleControl(), 2, vehicle_config)
    assert 0 < state.longitudinal_velocity < 50


def test_maximum_speed_limits_combined_velocity():
    config = VehicleConfig(engine_acceleration=100, maximum_speed=30)
    state = run(VehicleState(), VehicleControl(throttle=1), 10, config)
    assert state.speed <= 30+1e-10


@pytest.mark.parametrize("steering", [-.25, .25])
def test_steering_changes_heading_smoothly(steering, vehicle_config):
    state = VehicleState(longitudinal_velocity=12)
    integrate_vehicle(state, VehicleControl(steering=steering), vehicle_config, 1/120)
    assert 0 < abs(state.steering_angle) < abs(steering*vehicle_config.maximum_steering_angle)
    run(state, VehicleControl(steering=steering), 1, vehicle_config)
    assert state.heading*steering > 0


def test_stationary_steering_does_not_rotate_car(vehicle_config):
    state = run(VehicleState(), VehicleControl(steering=1), 2, vehicle_config)
    assert state.heading == 0
    assert state.speed == 0


def test_high_speed_cornering_slips_and_recovers(vehicle_config):
    state = run(VehicleState(longitudinal_velocity=50), VehicleControl(steering=.3), 2, vehicle_config)
    assert abs(state.lateral_velocity) > 2
    initial_slip = abs(state.lateral_velocity)
    run(state, VehicleControl(brake=.35), 4, vehicle_config)
    assert abs(state.lateral_velocity) < initial_slip*.2


def test_grass_reduces_available_cornering_force(vehicle_config):
    asphalt = run(VehicleState(longitudinal_velocity=45), VehicleControl(steering=.2), 1, vehicle_config)
    grass = run(VehicleState(longitudinal_velocity=45), VehicleControl(steering=.2), 1, vehicle_config, grip=.38)
    # Less grip reduces yaw authority as well as lateral force; body-frame slip
    # alone is not a reliable comparison when the two cars rotate differently.
    assert abs(grass.heading) < abs(asphalt.heading)*.8
    assert grass.y < asphalt.y*.8


def test_extreme_controls_remain_finite(vehicle_config):
    state = run(VehicleState(longitudinal_velocity=80, lateral_velocity=20), VehicleControl(steering=1, throttle=1), 30, vehicle_config)
    assert all(math.isfinite(getattr(state, name)) for name in ("x", "y", "heading", "yaw_rate", "lateral_velocity"))
    assert state.speed <= vehicle_config.maximum_speed+1e-9


def test_continuous_controls_clamp_and_reject_nonfinite():
    assert VehicleControl(steering=5, throttle=2, brake=-1) == VehicleControl(1, 1, 0)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            VehicleControl(steering=value)
        with pytest.raises(ValueError):
            VehicleConfig(mass=value)
