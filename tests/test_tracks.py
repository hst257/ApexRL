import json
import math
from pathlib import Path
import numpy as np
import pytest
from apexrl.core.track import Track


def test_three_tracks_have_connected_geometry(tracks):
    assert [track.id for track in tracks] == ["rookie_ring", "apex_valley", "velocity_circuit"]
    for track in tracks:
        assert len(track.points) > 500
        assert 0 < track.lengths[-1] < 2.1
        assert track.total_length == pytest.approx(track.lengths.sum())
        assert len(track.left_boundary) == len(track.points)
        widths = np.linalg.norm(track.left_boundary-track.right_boundary, axis=1)
        assert np.allclose(widths, track.widths)
        assert track.surface(*track.finish.center) == "asphalt"
        assert all(track.surface(*gate.center) == "asphalt" for gate in track.checkpoints)
        assert all(a.progress < b.progress for a, b in zip(track.checkpoints, track.checkpoints[1:]))


def test_grid_projection_matches_global_search(tracks):
    rng = np.random.default_rng(15)
    for track in tracks:
        x0, y0, x1, y1 = track.bounds
        for x, y in zip(rng.uniform(x0-150, x1+150, 50), rng.uniform(y0-150, y1+150, 50)):
            actual = track.project(x, y)
            expected = track._project_indices(x, y, np.arange(len(track.points)))
            assert actual.distance == pytest.approx(expected.distance, abs=1e-9)
            assert actual.progress == pytest.approx(expected.progress, abs=1e-8)


def test_progress_wraps_continuously(tracks):
    for track in tracks:
        a = track.project(*track.pose(track.total_length-.2)[:2]).progress
        b = track.project(*track.pose(.2)[:2]).progress
        difference = (b-a+track.total_length/2)%track.total_length-track.total_length/2
        assert difference == pytest.approx(.4, abs=.01)
        assert math.isfinite(track.curvature(a, 50))


def test_surface_bands_and_width_transition(tracks):
    track = tracks[2]
    for progress in (50, track.total_length*.5):
        x, y, angle = track.pose(progress)
        width = track.project(x, y).width
        nx, ny = -math.sin(angle), math.cos(angle)
        assert track.surface(x, y) == "asphalt"
        assert track.surface(x+nx*(width/2+.5), y+ny*(width/2+.5)) == "curb"
        assert track.surface(x+nx*(width/2+3), y+ny*(width/2+3)) == "grass"


@pytest.mark.parametrize("change", ["duplicate", "width", "checkpoint", "nonfinite", "crossing"])
def test_invalid_geometry_rejected(change):
    definition = json.loads((Path(__file__).resolve().parents[1]/"tracks/rookie_ring.json").read_text())
    if change == "duplicate":
        definition["control_points"][1] = definition["control_points"][0]
    elif change == "width":
        definition["road_width"] = -1
    elif change == "checkpoint":
        definition["checkpoint_fractions"] = [.5, .2]
    elif change == "nonfinite":
        definition["control_points"][0][0] = math.nan
    else:
        definition["control_points"] = [[0,0], [200,200], [0,200], [200,0]]
    with pytest.raises(ValueError):
        Track(definition)


def test_track_arrays_cannot_be_modified(tracks):
    with pytest.raises(ValueError):
        tracks[0].points[0, 0] = 99


def test_runoff_barriers_have_curvature_clearance(tracks):
    for track in tracks:
        assert np.max(np.abs(track.curvatures)*(track.widths/2+track.RUNOFF_WIDTH)) < .9


def test_folded_runoff_barrier_is_rejected():
    definition = json.loads((Path(__file__).resolve().parents[1]/"tracks/apex_valley.json").read_text())
    definition["control_points"][7] = [170,170]
    with pytest.raises(ValueError, match="runoff barriers"):
        Track(definition)
