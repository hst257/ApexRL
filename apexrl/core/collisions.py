"""Footprint-based swept collision checks against circuit runoff barriers."""
import math
from .track import Track
from .vehicle import VehicleConfig, VehicleState


def resolve_barriers(state: VehicleState, previous: VehicleState, track: Track, config: VehicleConfig,
                     previous_distance: float | None = None) -> bool:
    travel = math.hypot(state.x-previous.x, state.y-previous.y)
    angle = (state.heading-previous.heading+math.pi) % (2*math.pi)-math.pi
    if previous_distance is not None and previous_distance+travel+math.hypot(config.length, config.width)/2 < track.minimum_width/2+track.RUNOFF_WIDTH-0.1:
        return False
    samples = max(1, math.ceil((travel+abs(angle)*config.length/2)/0.5))
    for sample in range(1, samples+1):
        fraction = sample/samples
        probe = VehicleState(x=previous.x+(state.x-previous.x)*fraction,
                             y=previous.y+(state.y-previous.y)*fraction,
                             heading=previous.heading+angle*fraction)
        deepest = 0.0
        contact = None
        for x, y in probe.footprint(config):
            p = track.project(x, y)
            penetration = p.distance-(p.width/2+track.RUNOFF_WIDTH)
            if penetration > deepest:
                deepest, contact = penetration, p
        if contact is None:
            continue
        side = 1 if contact.lateral_distance >= 0 else -1
        nx, ny = -contact.tangent[1]*side, contact.tangent[0]*side
        state.x, state.y = probe.x-nx*(deepest+0.03), probe.y-ny*(deepest+0.03)
        state.heading = probe.heading
        wx, wy = state.world_velocity()
        outward = wx*nx+wy*ny
        if outward > 0:
            wx -= nx*outward*(1+config.collision_restitution)
            wy -= ny*outward*(1+config.collision_restitution)
        wx, wy = wx*config.collision_damping, wy*config.collision_damping
        c, s = math.cos(state.heading), math.sin(state.heading)
        state.longitudinal_velocity = wx*c+wy*s
        state.lateral_velocity = -wx*s+wy*c
        state.yaw_rate *= config.collision_damping
        return True
    return False
