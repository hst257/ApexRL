from dataclasses import dataclass, replace
import math
import random

from .collisions import resolve_barriers
from .controls import VehicleControl
from .physics import integrate_vehicle
from .timing import LapTiming, RaceEvent
from .track import Track
from .vehicle import VehicleConfig, VehicleState


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    countdown_seconds: float = 3.0
    shortcut_seconds: float = 2.0
    recovery_seconds: float = 8.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in (self.countdown_seconds, self.shortcut_seconds, self.recovery_seconds)):
            raise ValueError("Simulation durations must be finite and nonnegative")
        if self.shortcut_seconds <= 0:
            raise ValueError("shortcut_seconds must be positive")
        if self.recovery_seconds <= self.shortcut_seconds:
            raise ValueError("Recovery must follow shortcut invalidation")


@dataclass(frozen=True, slots=True)
class VehicleSnapshot:
    x: float
    y: float
    heading: float
    longitudinal_velocity: float
    lateral_velocity: float
    yaw_rate: float
    steering_angle: float
    steering_input: float
    throttle_input: float
    brake_input: float

    @property
    def speed(self) -> float:
        return math.hypot(self.longitudinal_velocity, self.lateral_velocity)


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    vehicle: VehicleSnapshot
    simulation_time: float
    countdown: float
    surface: str
    offtrack: bool
    offtrack_seconds: float
    lateral_distance: float
    track_progress: float
    next_checkpoint: int
    collision: bool
    lap_number: int
    completed_laps: int
    lap_time: float
    previous_lap: float | None
    best_lap: float | None
    lap_valid: bool
    invalid_reason: str


@dataclass(frozen=True, slots=True)
class StepResult:
    state: SimulationSnapshot
    events: tuple[RaceEvent, ...] = ()


class RacingSimulation:
    def __init__(self, track: Track, config: VehicleConfig | None = None, seed: int = 42,
                 simulation_config: SimulationConfig | None = None):
        self.track = track
        self.config = config or VehicleConfig()
        self.simulation_config = simulation_config or SimulationConfig()
        self.seed = seed
        self.random = random.Random(seed)
        self.timing = LapTiming(track)
        self.reset()

    def reset(self, *, seed: int | None = None, lateral_offset: float = 0.0,
              heading_offset: float = 0.0) -> SimulationSnapshot:
        """Start a fresh race with an optional validated grid-pose perturbation."""
        if not math.isfinite(lateral_offset) or not math.isfinite(heading_offset):
            raise ValueError("Reset offsets must be finite")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer")
        x, y, heading = self.track.pose(self.track.start_progress)
        x -= math.sin(heading)*lateral_offset
        y += math.cos(heading)*lateral_offset
        candidate = VehicleState(x=x, y=y, heading=(heading+heading_offset+math.pi) % (2*math.pi)-math.pi)
        if any(self.track.surface(*point) != "asphalt" for point in candidate.footprint(self.config)):
            raise ValueError("Reset footprint must fit entirely on asphalt")
        if seed is not None:
            self.seed = seed
        self.random.seed(self.seed)
        self.vehicle = candidate
        self.timing.reset()
        self.simulation_time = 0.0
        self.countdown = self.simulation_config.countdown_seconds
        self.offtrack_seconds = 0.0
        self.surface = "asphalt"
        self.offtrack = False
        self.collision = False
        self.projection = self.track.project(x, y)
        self.wheel_surfaces = tuple(self.track.surface(x, y) for x, y in self.vehicle.footprint(self.config, wheels=True))
        return self.snapshot()

    def snapshot(self) -> SimulationSnapshot:
        v, t = self.vehicle, self.timing
        car = VehicleSnapshot(v.x, v.y, v.heading, v.longitudinal_velocity, v.lateral_velocity,
                              v.yaw_rate, v.steering_angle, v.steering_input, v.throttle_input, v.brake_input)
        return SimulationSnapshot(car, self.simulation_time, self.countdown, self.surface, self.offtrack,
                                  self.offtrack_seconds, self.projection.lateral_distance, self.projection.progress,
                                  t.next_checkpoint, self.collision, t.lap_number, t.completed_laps,
                                  t.lap_time, t.previous_lap, t.best_lap, t.valid, t.invalid_reason)

    def recover_car(self) -> StepResult:
        checkpoint = self.timing.last_checkpoint
        progress = self.track.checkpoints[checkpoint].progress if checkpoint >= 0 else self.track.start_progress
        x, y, heading = self.track.pose(progress+4)
        self.vehicle = VehicleState(x=x, y=y, heading=heading)
        self.projection = self.track.project(x, y)
        self.offtrack_seconds, self.offtrack, self.collision = 0.0, False, False
        self.surface = "asphalt"
        self.wheel_surfaces = tuple(self.track.surface(x, y) for x, y in self.vehicle.footprint(self.config, wheels=True))
        events = self.timing.invalidate("Car recovered")+(RaceEvent("recovered", self.simulation_time),)
        return StepResult(self.snapshot(), events)

    def step(self, action: VehicleControl, dt: float = 1/120) -> StepResult:
        if not isinstance(action, VehicleControl):
            raise TypeError("action must be VehicleControl")
        if not math.isfinite(dt) or not 0 < dt <= 1/60:
            raise ValueError("dt must be finite and in (0, 1/60]")
        self.simulation_time += dt
        if self.countdown > 1e-10:
            self.countdown = max(0.0, self.countdown-dt)
            if self.countdown < 1e-10:
                self.countdown = 0.0
            return StepResult(self.snapshot())
        previous = replace(self.vehicle)
        old_progress = self.projection.progress
        surfaces = self.wheel_surfaces
        grass_fraction = surfaces.count("grass")/4
        curb_fraction = surfaces.count("curb")/4
        grip = 1-grass_fraction*(1-self.config.offtrack_traction)-curb_fraction*(1-self.config.curb_grip)
        rolling = 1+grass_fraction*(self.config.grass_rolling_resistance/self.config.rolling_resistance-1) if self.config.rolling_resistance else 1
        engine = 1-grass_fraction*(1-self.config.offtrack_traction)
        integrate_vehicle(self.vehicle, action, self.config, dt, grip, rolling, engine)
        self.collision = resolve_barriers(self.vehicle, previous, self.track, self.config, self.projection.distance)
        self.projection = self.track.project(self.vehicle.x, self.vehicle.y)
        current_surfaces = tuple(self.track.surface(x, y) for x, y in self.vehicle.footprint(self.config, wheels=True))
        self.wheel_surfaces = current_surfaces
        self.offtrack = "grass" in current_surfaces
        all_grass = all(s == "grass" for s in current_surfaces)
        self.offtrack_seconds = self.offtrack_seconds+dt if all_grass else 0.0
        self.surface = "grass" if self.offtrack else "curb" if "curb" in current_surfaces else "asphalt"
        events = self.timing.update((previous.x, previous.y), (self.vehicle.x, self.vehicle.y),
                                    old_progress, self.projection.progress, dt)
        if self.collision:
            events += (RaceEvent("collision", self.simulation_time),)
        if self.offtrack_seconds >= self.simulation_config.shortcut_seconds:
            events += self.timing.invalidate("Extended off-track driving")
        if self.offtrack_seconds >= self.simulation_config.recovery_seconds:
            recovery = self.recover_car()
            events += recovery.events
        return StepResult(self.snapshot(), events)
