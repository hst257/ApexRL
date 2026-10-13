import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import Future
import pygame
import pytest

from apexrl.core import RacingSimulation, VehicleControl
from apexrl.core.simulation import SimulationConfig
from apexrl.game.application import Application
from apexrl.game.policies import discover_policies

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    application = Application(ROOT, tmp_path/"saves")
    yield application
    application.close()


class FakeSession:
    def __init__(self, track, vehicle):
        self.simulation = RacingSimulation(track, vehicle, simulation_config=SimulationConfig(countdown_seconds=0))
        self.step_seconds = 4/120
        self.reset()

    def reset(self):
        self.simulation.reset()
        self.finished = False
        self.info = {"max_progress_fraction": 0, "physics_ticks": 4}

    def step(self):
        for _ in range(4):
            self.simulation.step(VehicleControl(throttle=.8))
        return self.info

    def close(self):
        pass


def test_policy_discovery_ignores_bad_or_unpaired_metadata(tmp_path):
    run = tmp_path/"runs"/"test"
    run.mkdir(parents=True)
    (run/"best_model.zip").write_bytes(b"test")
    metadata = {"num_timesteps": 100, "format_version": 1, "training_track": "rookie_ring", "promotion": None}
    (run/"best_model.metadata.json").write_text(json.dumps(metadata))
    (run/"missing.metadata.json").write_text(json.dumps(metadata))
    (run/"broken.metadata.json").write_text("{bad")
    (run/"wrong.metadata.json").write_text("[]")
    (run/"wrong.zip").write_bytes(b"test")
    entries = discover_policies(tmp_path)
    assert len(entries) == 1
    assert entries[0].timesteps == 100
    assert entries[0].path == run/"best_model.zip"


def test_ai_garage_missing_dependencies_is_actionable(app, monkeypatch):
    monkeypatch.setattr("apexrl.game.application.rl_available", lambda: False)
    app.activate("ai")
    app.draw()
    assert app.screen == "ai"
    assert not app.start_ai()
    assert "requirements-rl.txt" in app.ai_error
    app.activate("back")
    assert app.screen == "main"


def test_published_policies_sort_first_and_report_circuit_validation(tmp_path):
    directory = tmp_path/"models"
    directory.mkdir()
    data = {"format_version": 1, "training_track": "rookie_ring", "num_timesteps": 50000,
            "source_timesteps": 480032,
            "promotion": {"summary": {"rookie_ring": {"episodes":20, "success_rate":1.0}}}}
    (directory/"driver.zip").write_bytes(b"model")
    (directory/"driver.metadata.json").write_text(json.dumps(data))
    run = tmp_path/"runs"/"later"
    run.mkdir(parents=True)
    (run/"final.zip").write_bytes(b"model")
    (run/"final.metadata.json").write_text(json.dumps(dict(data, num_timesteps=1000000)))
    entries = discover_policies(tmp_path)
    assert entries[0].path == directory/"driver.zip"
    assert entries[0].validation == (("rookie_ring",20,1.0),)
    assert entries[0].source_timesteps == 480032


def test_garage_prefers_driver_validated_on_more_circuits(tmp_path):
    directory = tmp_path/"models"
    directory.mkdir()
    for name, steps, circuits in (("specialist", 1000000, ("rookie_ring",)),
                                  ("all-circuits", 100000, ("rookie_ring", "apex_valley", "velocity_circuit"))):
        data = {"format_version": 1, "training_track": "all", "num_timesteps": steps,
                "promotion": {"summary": {track: {"episodes":20, "success_rate":1.0} for track in circuits}},
                "source_bootstrap": {"demonstration_transitions":23425}}
        (directory/f"{name}.zip").write_bytes(b"model")
        (directory/f"{name}.metadata.json").write_text(json.dumps(data))
    assert discover_policies(tmp_path)[0].path.name == "all-circuits.zip"
    assert discover_policies(tmp_path)[0].demonstration_steps == 23425


def test_ai_pause_restart_and_manual_records_are_isolated(app):
    app.records.record(app.tracks[0], 40)
    before = dict(app.records.records)
    app.ai_session = FakeSession(app.tracks[0], app.vehicle_config)
    app.camera.reset(app.ai_session.simulation.snapshot().vehicle)
    app.transition("ai_race")
    app.advance(1/30)
    assert app.ai_session.simulation.timing.elapsed == pytest.approx(1/30)
    snapshot = app.ai_session.simulation.snapshot()
    app.event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    app.advance(.5)
    app.draw()
    assert app.screen == "ai_pause"
    assert app.ai_session.simulation.snapshot() == snapshot
    app.event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    app.event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    assert app.ai_session.simulation.timing.elapsed == 0
    track = app.tracks[0]
    sim = app.ai_session.simulation
    x, y, heading = track.pose(track.total_length-.05)
    sim.vehicle.x, sim.vehicle.y, sim.vehicle.heading = x, y, heading
    sim.vehicle.longitudinal_velocity = 12
    sim.projection = track.project(x, y)
    sim.timing.next_checkpoint = len(track.checkpoints)
    sim.timing.forward_distance = track.total_length-.05
    sim.timing.finish_armed = True
    sim.timing.elapsed = 30
    app.advance(1/30)
    assert sim.snapshot().completed_laps == 1
    assert app.records.records == before
    app.toast_seconds = 3600
    app.activate("ai")
    assert app.toast_seconds == 0
    app.start_race()
    assert app.ai_session is None and app.screen == "race"


def test_manual_application_import_never_loads_rl_dependencies():
    code = "import sys; sys.modules['torch']=None; sys.modules['gymnasium']=None; sys.modules['stable_baselines3']=None; from apexrl.game.application import Application"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class ControlledLoader:
    def __init__(self):
        self.future = Future()

    def submit(self, *args):
        return self.future

    def shutdown(self, **kwargs):
        pass


def prepare_async_load(app, monkeypatch):
    from apexrl.game.policies import PolicyEntry
    monkeypatch.setattr("apexrl.game.application.rl_available", lambda: True)
    app.ai_models = [PolicyEntry(Path("policy.zip"), "Policy", "all", 1)]
    app.ai_loader = ControlledLoader()
    app.request_ai()
    return app.ai_loader.future


def test_loading_keeps_menu_responsive_and_respects_lost_focus(app, monkeypatch):
    future = prepare_async_load(app, monkeypatch)
    assert app.screen == "ai_loading"
    app.advance(.5)
    app.draw()
    assert app.menus.selected_action() == "ai:cancel"
    app.event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    session = FakeSession(app.tracks[0], app.vehicle_config)
    future.set_result(session)
    app.advance(0)
    assert app.ai_session is session and app.screen == "ai_pause"
    assert session.simulation.timing.elapsed == 0


def test_cancelled_loading_closes_completed_session_without_starting_race(app, monkeypatch):
    future = prepare_async_load(app, monkeypatch)
    future.set_running_or_notify_cancel()
    session = FakeSession(app.tracks[0], app.vehicle_config)
    closed = []
    session.close = lambda: closed.append(True)
    app.activate("ai:cancel")
    future.set_result(session)
    app.advance(.5)
    assert app.screen == "ai" and app.ai_session is None and closed == [True]


def test_loading_error_returns_to_garage_with_actionable_message(app, monkeypatch):
    future = prepare_async_load(app, monkeypatch)
    future.set_exception(ValueError("Checkpoint physics differs"))
    app.advance(0)
    app.draw()
    assert app.screen == "ai" and "physics differs" in app.ai_error


def test_refresh_keeps_selected_published_driver_and_can_show_training(app, monkeypatch):
    from apexrl.game.policies import PolicyEntry
    first = PolicyEntry(app.root/"models/a.zip", "A", "all", 2)
    second = PolicyEntry(app.root/"models/b.zip", "B", "rookie_ring", 3)
    training = PolicyEntry(app.root/"runs/final.zip", "Run", "all", 1000)
    monkeypatch.setattr("apexrl.game.application.discover_policies", lambda root: [first,second,training])
    app.refresh_policies()
    assert app.ai_models == [first,second]
    app.ai_model_index = 1
    app.refresh_policies()
    assert app.ai_models[app.ai_model_index] == second
    app.activate("ai:library")
    assert len(app.ai_models) == 3 and app.ai_models[app.ai_model_index] == second
