import math
import numpy as np

BASE_NAMES = (
    "forward_speed", "lateral_speed", "yaw_rate", "heading_sin", "heading_cos",
    "lateral_offset", "steering_angle", "steering_command", "throttle", "brake",
    "road_width", "asphalt_wheels", "curb_wheels", "grass_wheels", "next_gate_distance",
    "progress_fraction", "time_remaining", "stall_remaining", "offtrack_remaining", "wrong_way_remaining",
)
OBSERVATION_VERSION = 1
ACTION_VERSION = 1


def observation_names(config) -> tuple[str, ...]:
    return BASE_NAMES+tuple(f"lookahead_{distance:g}m_{name}" for distance in config.lookahead_distances
                            for name in ("forward", "lateral", "heading_sin", "heading_cos", "curvature"))


def observe(simulation, config, stalled_seconds: float, wrong_way_seconds: float = 0.0) -> np.ndarray:
    vehicle, track, p = simulation.vehicle, simulation.track, simulation.projection
    angle = math.atan2(p.tangent[1], p.tangent[0])-vehicle.heading
    gate = track.checkpoints[simulation.timing.next_checkpoint] if simulation.timing.next_checkpoint < len(track.checkpoints) else track.finish
    max_speed = simulation.config.maximum_speed
    surfaces = simulation.wheel_surfaces
    values = [vehicle.longitudinal_velocity/max_speed, vehicle.lateral_velocity/max_speed, vehicle.yaw_rate/3,
              math.sin(angle), math.cos(angle), p.lateral_distance/(p.width/2+track.RUNOFF_WIDTH),
              vehicle.steering_angle/simulation.config.maximum_steering_angle if simulation.config.maximum_steering_angle else 0,
              vehicle.steering_input, vehicle.throttle_input, vehicle.brake_input, p.width/20,
              surfaces.count("asphalt")/4, surfaces.count("curb")/4, surfaces.count("grass")/4,
              (gate.progress-p.progress) % track.total_length/track.total_length,
              (p.progress-track.start_progress) % track.total_length/track.total_length,
              1-simulation.timing.elapsed/config.max_episode_seconds, 1-stalled_seconds/config.stall_seconds,
              1-simulation.offtrack_seconds/config.offtrack_seconds, 1-wrong_way_seconds/config.wrong_way_seconds]
    c, s = math.cos(vehicle.heading), math.sin(vehicle.heading)
    for distance in config.lookahead_distances:
        progress = (p.progress+distance) % track.total_length
        x, y, heading = track.pose(progress)
        dx, dy = x-vehicle.x, y-vehicle.y
        scale = distance+p.width/2+track.RUNOFF_WIDTH
        index = min(len(track.curvatures)-1, int(np.searchsorted(track.cumulative, progress, side="right"))-1)
        values.extend(((dx*c+dy*s)/scale, (-dx*s+dy*c)/scale,
                       math.sin(heading-vehicle.heading), math.cos(heading-vehicle.heading), track.curvatures[index]*25))
    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)
