import math
import pytest
from apexrl.core.timing import LapTiming
from apexrl.core.geometry import gate_crossing


def walk_lap(timing, lateral_offset=0):
    track = timing.track
    previous = track.pose(0)[:2]
    old_progress = 0
    events = []
    for progress in list(track.cumulative[1:-1])+[track.total_length]:
        x, y, heading = track.pose(progress)
        current = (x-math.sin(heading)*lateral_offset, y+math.cos(heading)*lateral_offset)
        events.extend(timing.update(previous, current, old_progress, progress % track.total_length, .1))
        previous, old_progress = current, progress % track.total_length
    return events


def test_valid_circuit_counts_and_records_time(tracks):
    for track in tracks:
        timing = LapTiming(track)
        events = walk_lap(timing)
        assert timing.completed_laps == 1
        assert timing.lap_number == 2
        assert timing.best_lap > 0
        assert any(event.kind == "lap_completed" for event in events)


def test_finish_oscillation_cannot_count_without_checkpoints(tracks):
    timing = LapTiming(tracks[0])
    gate = tracks[0].finish
    a = tuple(gate.center[i]-gate.tangent[i]*2 for i in (0,1))
    b = tuple(gate.center[i]+gate.tangent[i]*2 for i in (0,1))
    for _ in range(20):
        timing.update(a, b, tracks[0].total_length-2, 2, .1)
        timing.update(b, a, 2, tracks[0].total_length-2, .1)
    assert timing.completed_laps == 0
    assert timing.lap_number == 1


def test_skipping_checkpoint_invalidates(tracks):
    timing = LapTiming(tracks[0])
    gate = tracks[0].checkpoints[1]
    a = tuple(gate.center[i]-gate.tangent[i]*2 for i in (0,1))
    b = tuple(gate.center[i]+gate.tangent[i]*2 for i in (0,1))
    timing.update(a, b, gate.progress-2, gate.progress+2, .1)
    assert not timing.valid
    assert timing.next_checkpoint == 0


def test_reverse_and_outside_gate_crossings_ignored():
    assert gate_crossing((1,0), (-1,0), (0,0), (1,0), 5) is None
    assert gate_crossing((-1,6), (1,6), (0,0), (1,0), 5) is None
    assert gate_crossing((-1,0), (3,0), (0,0), (1,0), 5) == pytest.approx(.25)


def test_invalid_attempt_starts_fresh_at_finish(tracks):
    timing = LapTiming(tracks[0])
    timing.invalidate("Car recovered")
    walk_lap(timing)
    assert timing.completed_laps == 0
    assert timing.lap_number == 2
    assert timing.valid
    assert timing.lap_time < .101
    walk_lap(timing)
    assert timing.completed_laps == 1


def test_progress_jump_invalidates_even_before_gate(tracks):
    timing = LapTiming(tracks[0])
    timing.update((0,0), (1,0), 0, 150, .01)
    assert not timing.valid


def test_crossing_time_interpolates_within_step(tracks):
    timing = LapTiming(tracks[0])
    timing.next_checkpoint = len(tracks[0].checkpoints)
    timing.forward_distance = tracks[0].total_length-1
    timing.finish_armed = True
    gate = tracks[0].finish
    a = tuple(gate.center[i]-gate.tangent[i] for i in (0,1))
    b = tuple(gate.center[i]+gate.tangent[i]*3 for i in (0,1))
    timing.elapsed = 50
    timing.update(a, b, tracks[0].total_length-1, 3, .01)
    assert timing.previous_lap == pytest.approx(50.0025)
