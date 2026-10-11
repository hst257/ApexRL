"""Gymnasium adapter. Rendering is lazy; default stepping never imports Pygame."""
import math
from pathlib import Path
import gymnasium as gym
import numpy as np

from apexrl.core import RacingSimulation, VehicleConfig, VehicleControl, load_tracks
from apexrl.core.simulation import SimulationConfig
from .config import EnvConfig
from .observations import observe, observation_names

ROOT = Path(__file__).resolve().parents[2]
TRACK_IDS = ("rookie_ring", "apex_valley", "velocity_circuit")


class ApexRacingEnv(gym.Env):
    """Actions: [steering, signed pedal], both in [-1, 1]. Positive pedal is throttle."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, track_id="rookie_ring", *, config: EnvConfig | None = None,
                 vehicle_config: VehicleConfig | None = None, root: str | Path = ROOT,
                 render_mode: str | None = None, tracks=None):
        super().__init__()
        if track_id not in (*TRACK_IDS, "all"):
            raise ValueError(f"Unknown track: {track_id}")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.root = Path(root)
        self.tracks = tracks if tracks is not None else load_tracks(self.root/"tracks")
        if {track.id for track in self.tracks} != set(TRACK_IDS):
            raise ValueError("Supply all three original tracks")
        self.track_id = track_id
        self.config = config or EnvConfig()
        self.vehicle_config = vehicle_config or VehicleConfig.load(self.root/"config/vehicle.json")
        self.render_mode = render_mode
        self.metadata = dict(self.metadata, render_fps=round(120/self.config.action_repeat))
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_names = observation_names(self.config)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(len(self.observation_names),), dtype=np.float32)
        self.simulation = None
        self._viewer = None
        self._needs_reset = True
        self.last_info = {}

    def reset(self, *, seed=None, options=None):
        options = options or {}
        unknown = set(options)-{"track_id", "evaluation"}
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        requested = options.get("track_id", self.track_id)
        if requested not in (*TRACK_IDS, "all"):
            raise ValueError(f"Unknown track: {requested}")
        if not isinstance(options.get("evaluation", False), bool):
            raise ValueError("evaluation must be a boolean")
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        if requested == "all":
            index = int(self.np_random.integers(len(self.tracks)))
        else:
            index = next(i for i, track in enumerate(self.tracks) if track.id == requested)
        core_seed = int(self.np_random.integers(0, 2**31-1))
        self.simulation = RacingSimulation(self.tracks[index], self.vehicle_config, seed=core_seed,
                                           simulation_config=SimulationConfig(countdown_seconds=0))
        evaluation = options.get("evaluation", False)
        lateral = 0.0 if evaluation else float(self.np_random.uniform(-self.config.spawn_lateral_jitter, self.config.spawn_lateral_jitter))
        heading = 0.0 if evaluation else float(self.np_random.uniform(-self.config.spawn_heading_jitter, self.config.spawn_heading_jitter))
        self.simulation.reset(seed=core_seed, lateral_offset=lateral, heading_offset=heading)
        self.episode_return = self.progress = self.best_progress = self.max_progress = 0.0
        self.stalled_seconds = self.wrong_way_seconds = 0.0
        self._previous_steering = 0.0
        self._needs_reset = False
        self.last_info = self._info({}, (), None, False, 0)
        if self._viewer:
            self._viewer.reset(self.simulation.snapshot().vehicle)
        if self.render_mode == "human":
            self.render()
        return self._observation(), dict(self.last_info)

    def _observation(self):
        return observe(self.simulation, self.config, self.stalled_seconds, self.wrong_way_seconds)

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping, including after an episode ends")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must contain two finite values: steering, signed pedal")
        steering, pedal = np.clip(action, -1, 1)
        control = VehicleControl(float(steering), max(0.0, float(pedal)), max(0.0, -float(pedal)))
        weights = self.config.reward
        terms = {"progress": 0.0, "time": 0.0, "grass": 0.0, "slip": 0.0,
                 "steering_change": -weights.steering_change*(float(steering)-self._previous_steering)**2,
                 "checkpoint": 0.0, "lap": 0.0, "failure": 0.0}
        reason = None
        terminated = truncated = False
        events = []
        ticks = 0
        dt = 1/120
        for _ in range(self.config.action_repeat):
            before = self.simulation.snapshot()
            result = self.simulation.step(control, dt)
            state = result.state
            ticks += 1
            events.extend(result.events)
            kinds = {event.kind for event in result.events}
            lap_complete = "lap_completed" in kinds
            invalid = not state.lap_valid or bool(kinds & {"lap_invalidated", "invalid_lap_finished", "recovered"})
            delta = (state.track_progress-before.track_progress+self.simulation.track.total_length/2) % self.simulation.track.total_length-self.simulation.track.total_length/2
            travel = math.hypot(state.vehicle.x-before.vehicle.x, state.vehicle.y-before.vehicle.y)
            if invalid or abs(delta) > 2*travel+5:
                delta = 0.0  # Never pay for a reset teleport or projection exploit.
            self.progress += delta
            self.max_progress = max(self.max_progress, self.progress)
            self.stalled_seconds += dt
            if self.progress >= self.best_progress+self.config.progress_threshold:
                self.best_progress = self.progress
                self.stalled_seconds = 0.0
            self.wrong_way_seconds = self.wrong_way_seconds+dt if delta < -.1*dt else max(0.0, self.wrong_way_seconds-dt)
            terms["progress"] += delta*weights.progress_per_meter
            terms["time"] -= dt*weights.time_per_second
            terms["grass"] -= dt*weights.grass_per_second*self.simulation.wheel_surfaces.count("grass")/4
            terms["slip"] -= dt*weights.slip_per_second*abs(state.vehicle.lateral_velocity)/self.vehicle_config.maximum_speed
            terms["checkpoint"] += weights.checkpoint*sum(event.kind == "checkpoint" for event in result.events)
            if lap_complete:
                reason, terminated = "lap_completed", True
                terms["lap"] += weights.lap
            elif state.collision:
                reason, terminated = "collision", True
            elif invalid:
                reason, terminated = "invalid_lap", True
            elif state.offtrack_seconds+1e-9 >= self.config.offtrack_seconds:
                reason, terminated = "offtrack", True
            elif self.wrong_way_seconds+1e-9 >= self.config.wrong_way_seconds:
                reason, terminated = "wrong_way", True
            elif self.stalled_seconds+1e-9 >= self.config.stall_seconds:
                reason, terminated = "stalled", True
            elif self.simulation.timing.elapsed+1e-9 >= self.config.max_episode_seconds:
                reason, truncated = "time_limit", True
            if terminated or truncated:
                break
        if terminated and reason != "lap_completed":
            terms["failure"] -= weights.failure
        reward = float(sum(terms.values()))
        self.episode_return += reward
        self._previous_steering = float(steering)
        self._needs_reset = terminated or truncated
        self.last_info = self._info(terms, events, reason, reason == "lap_completed", ticks)
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, bool(terminated), bool(truncated), dict(self.last_info)

    def _info(self, terms, events, reason, success, ticks):
        state = self.simulation.snapshot()
        return {"track_id": self.simulation.track.id, "track_version": self.simulation.track.version,
                "progress_m": self.progress, "progress_fraction": max(0.0, min(1.0, self.progress/self.simulation.track.total_length)),
                "max_progress_fraction": max(0.0, min(1.0, self.max_progress/self.simulation.track.total_length)),
                "episode_seconds": self.simulation.timing.elapsed, "episode_return": self.episode_return,
                "lap_time": state.previous_lap if success else None, "is_success": bool(success),
                "termination_reason": reason, "reward_terms": dict(terms), "physics_ticks": ticks,
                "events": tuple(event.kind for event in events), "speed_kmh": state.vehicle.speed*3.6,
                "checkpoint": state.next_checkpoint, "surface": state.surface, "collision": state.collision}

    def render(self):
        if self.render_mode is None:
            return None
        if self.simulation is None:
            raise gym.error.ResetNeeded("Call reset() before rendering")
        if self._viewer is None:
            from .viewer import PolicyViewer
            self._viewer = PolicyViewer(self.render_mode, self.root, self.vehicle_config)
            self._viewer.reset(self.simulation.snapshot().vehicle)
        return self._viewer.draw(self.simulation, self.last_info)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
