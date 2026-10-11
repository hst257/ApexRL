from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class VehicleControl:
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    def __post_init__(self) -> None:
        for name, lower, upper in (("steering", -1.0, 1.0), ("throttle", 0.0, 1.0), ("brake", 0.0, 1.0)):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, max(lower, min(upper, value)))
