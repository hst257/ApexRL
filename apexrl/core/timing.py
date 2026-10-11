from dataclasses import dataclass
import math
from .geometry import gate_crossing
from .track import Track


@dataclass(frozen=True, slots=True)
class RaceEvent:
    kind: str
    time: float = 0.0
    detail: str = ""


class LapTiming:
    def __init__(self, track: Track):
        self.track = track
        self.reset()

    def reset(self) -> None:
        self.elapsed = 0.0
        self.lap_start = 0.0
        self.lap_number = 1
        self.completed_laps = 0
        self.next_checkpoint = 0
        self.last_checkpoint = -1
        self.valid = True
        self.invalid_reason = ""
        self.previous_lap = None
        self.best_lap = None
        self.forward_distance = 0.0
        self.finish_armed = False

    @property
    def lap_time(self) -> float:
        return max(0.0, self.elapsed-self.lap_start)

    def invalidate(self, reason: str) -> tuple[RaceEvent, ...]:
        if not self.valid:
            return ()
        self.valid, self.invalid_reason = False, reason
        return (RaceEvent("lap_invalidated", self.elapsed, reason),)

    def update(self, previous, current, old_progress: float, new_progress: float, dt: float) -> tuple[RaceEvent, ...]:
        events = []
        movement = math.hypot(current[0]-previous[0], current[1]-previous[1])
        delta = (new_progress-old_progress+self.track.total_length/2) % self.track.total_length-self.track.total_length/2
        if abs(delta) > movement*2+5:
            events.extend(self.invalidate("Progress discontinuity"))
        else:
            self.forward_distance = max(0.0, self.forward_distance+delta)
        if self.forward_distance > self.track.total_length*0.1 or not self.valid:
            self.finish_armed = True
        crossings = []
        for index, gate in enumerate(self.track.checkpoints):
            fraction = gate_crossing(previous, current, gate.center, gate.tangent, gate.half_width)
            if fraction is not None:
                crossings.append((fraction, index))
        finish = self.track.finish
        fraction = gate_crossing(previous, current, finish.center, finish.tangent, finish.half_width)
        if fraction is not None:
            crossings.append((fraction, -1))
        for fraction, index in sorted(crossings):
            crossing_time = self.elapsed+fraction*dt
            if index >= 0:
                if index == self.next_checkpoint:
                    self.next_checkpoint += 1
                    self.last_checkpoint = index
                    events.append(RaceEvent("checkpoint", crossing_time, str(index+1)))
                elif index > self.next_checkpoint:
                    events.extend(self.invalidate("Checkpoint skipped"))
                continue
            if not self.finish_armed:
                continue
            valid_lap = self.valid and self.next_checkpoint == len(self.track.checkpoints) and self.forward_distance >= self.track.total_length*0.95
            lap_time = crossing_time-self.lap_start
            if valid_lap:
                self.completed_laps += 1
                self.previous_lap = lap_time
                if self.best_lap is None or lap_time < self.best_lap:
                    self.best_lap = lap_time
                    events.append(RaceEvent("best_lap", lap_time))
                events.append(RaceEvent("lap_completed", lap_time))
            else:
                events.append(RaceEvent("invalid_lap_finished", lap_time))
            self.lap_number += 1
            self.lap_start = crossing_time
            self.next_checkpoint, self.last_checkpoint = 0, -1
            self.valid, self.invalid_reason = True, ""
            self.forward_distance = 0.0
            self.finish_armed = False
        self.elapsed += dt
        return tuple(events)
