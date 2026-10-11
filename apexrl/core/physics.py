"""Reduced dynamic bicycle model; semi-implicit integration in body coordinates."""
import math

from .controls import VehicleControl
from .vehicle import VehicleConfig, VehicleState

GRAVITY = 9.81


def integrate_vehicle(state: VehicleState, control: VehicleControl, config: VehicleConfig,
                      dt: float, grip: float = 1.0, rolling: float = 1.0,
                      engine: float = 1.0) -> None:
    state.steering_input, state.throttle_input, state.brake_input = control.steering, control.throttle, control.brake
    target = control.steering * config.maximum_steering_angle
    state.steering_angle += (target - state.steering_angle) * (1 - math.exp(-config.steering_response * dt))
    u, v, r = state.longitudinal_velocity, state.lateral_velocity, state.yaw_rate
    sign = 1.0 if u >= 0 else -1.0
    resistance = config.rolling_resistance * rolling + config.aerodynamic_drag * u * u
    drive = control.throttle * config.engine_acceleration * engine
    ax = drive - sign * (control.brake * config.braking_deceleration + resistance)
    if abs(u) < 0.15 and drive <= control.brake * config.braking_deceleration + resistance:
        ax, u = 0.0, 0.0
    half = config.wheelbase / 2
    denominator = max(abs(u), 2.0)
    alpha_front = math.atan2(v + half * r, denominator) - state.steering_angle * sign
    alpha_rear = math.atan2(v - half * r, denominator)
    budget = config.tire_grip * grip * GRAVITY * config.mass / 2
    # Braking/throttle consume part of the combined tire-force budget.
    longitudinal_force = min(abs(ax) * config.mass / 2, budget * 0.96)
    lateral_limit = math.sqrt(max(0.0, budget * budget - longitudinal_force * longitudinal_force))
    front = max(-lateral_limit, min(lateral_limit, -config.front_stiffness * alpha_front))
    rear = max(-lateral_limit, min(lateral_limit, -config.rear_stiffness * alpha_rear))
    blend = min(1.0, max(0.0, (abs(u) - 2.0) / config.low_speed_transition))
    dynamic_r = r + (front * math.cos(state.steering_angle) - rear) * half / config.yaw_inertia * dt
    kinematic_r = u / config.wheelbase * math.tan(state.steering_angle)
    r = blend * dynamic_r + (1 - blend) * kinematic_r
    dynamic_v = v + ((front * math.cos(state.steering_angle) + rear) / config.mass - u * r) * dt
    v = blend * dynamic_v + (1 - blend) * v * math.exp(-8.0 * dt)
    v *= math.exp(-config.lateral_damping * grip * dt)
    new_u = u + (ax + v * r * blend) * dt
    if u >= 0 and new_u < 0:
        new_u = 0.0
    new_u = max(-config.maximum_speed * 0.1, min(config.maximum_speed, new_u))
    speed = math.hypot(new_u, v)
    if speed > config.maximum_speed:
        scale = config.maximum_speed / speed
        new_u, v = new_u * scale, v * scale
    # Finite bounds protect contact resolution and extreme sustained slides.
    r = max(-3.0, min(3.0, r))
    state.longitudinal_velocity, state.lateral_velocity, state.yaw_rate = new_u, v, r
    state.heading = (state.heading + r * dt + math.pi) % (2 * math.pi) - math.pi
    wx, wy = state.world_velocity()
    state.x += wx * dt
    state.y += wy * dt
