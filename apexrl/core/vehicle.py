from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VehicleConfig:
    mass: float = 800.0
    engine_acceleration: float = 12.0
    braking_deceleration: float = 22.0
    maximum_speed: float = 83.3333333333
    rolling_resistance: float = 0.22
    aerodynamic_drag: float = 0.0015
    steering_response: float = 4.5
    maximum_steering_angle: float = 0.46
    wheelbase: float = 3.4
    tire_grip: float = 1.65
    lateral_damping: float = 0.65
    offtrack_traction: float = 0.38
    length: float = 5.2
    width: float = 2.0
    yaw_inertia: float = 1800.0
    front_stiffness: float = 52000.0
    rear_stiffness: float = 58000.0
    low_speed_transition: float = 8.0
    grass_rolling_resistance: float = 2.3
    curb_grip: float = 0.88
    collision_restitution: float = 0.05
    collision_damping: float = 0.55

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid vehicle parameter: {field.name}")
        for name in ("mass", "wheelbase", "yaw_inertia", "low_speed_transition", "maximum_speed", "length", "width", "steering_response"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_steering_angle >= math.pi / 2 or self.wheelbase >= self.length:
            raise ValueError("Invalid steering angle or wheelbase")
        if not 0 < self.offtrack_traction <= 1 or not 0 < self.curb_grip <= 1:
            raise ValueError("Surface grip multipliers must be in (0, 1]")

    @classmethod
    def load(cls, path: str | Path) -> "VehicleConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    longitudinal_velocity: float = 0.0
    lateral_velocity: float = 0.0
    yaw_rate: float = 0.0
    steering_angle: float = 0.0
    steering_input: float = 0.0
    throttle_input: float = 0.0
    brake_input: float = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.longitudinal_velocity, self.lateral_velocity)

    def world_velocity(self) -> tuple[float, float]:
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (c * self.longitudinal_velocity - s * self.lateral_velocity,
                s * self.longitudinal_velocity + c * self.lateral_velocity)

    def footprint(self, config: VehicleConfig, wheels: bool = False) -> tuple[tuple[float, float], ...]:
        a = config.wheelbase / 2 if wheels else config.length / 2
        b = config.width / 2
        c, s = math.cos(self.heading), math.sin(self.heading)
        return tuple((self.x + c * u - s * v, self.y + s * u + c * v)
                     for u, v in ((a, b), (a, -b), (-a, -b), (-a, b)))
