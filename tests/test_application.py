from pathlib import Path
from collections import defaultdict
import pygame
import pytest
from apexrl.core.controls import VehicleControl
from apexrl.game.application import Application
from apexrl.game.input_handler import KeyboardController


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    application = Application(Path(__file__).resolve().parents[1], tmp_path)
    yield application
    application.close()


def key(app, value):
    app.event(pygame.event.Event(pygame.KEYDOWN, key=value))


def test_menu_keyboard_mouse_and_settings_navigation(app):
    key(app, pygame.K_DOWN)
    key(app, pygame.K_RETURN)
    assert app.screen == "tracks"
    app.draw()
    key(app, pygame.K_DOWN)
    key(app, pygame.K_RETURN)
    assert app.screen == "race" and app.current_track == 1
    key(app, pygame.K_ESCAPE)
    assert app.screen == "pause"
    app.draw()
    key(app, pygame.K_DOWN)
    key(app, pygame.K_DOWN)
    key(app, pygame.K_DOWN)
    key(app, pygame.K_RETURN)
    assert app.screen == "settings"
    app.draw()
    old = app.settings["volume"]
    key(app, pygame.K_RIGHT)
    assert app.settings["volume"] > old
    key(app, pygame.K_ESCAPE)
    assert app.screen == "pause"
    key(app, pygame.K_ESCAPE)
    assert app.screen == "race"
    app.transition("main")
    app.draw()
    app.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(100, 450), button=1))
    assert app.screen == "controls"


def test_pause_focus_loss_and_recovery_freeze_or_preserve_clock(app):
    app.start_race()
    for _ in range(220):
        app.advance(1/60, VehicleControl(throttle=1))
    assert app.state.vehicle.speed > 0
    before = app.state
    app.event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    app.advance(.5)
    assert app.state == before
    assert app.accumulator == 0
    key(app, pygame.K_ESCAPE)
    key(app, pygame.K_r)
    assert not app.state.lap_valid
    assert app.state.lap_time == before.lap_time
    assert app.state.vehicle.speed == 0


def test_resize_minimap_debug_and_each_track_render(app):
    for index in range(3):
        app.start_race(index)
        app.advance(1/60, VehicleControl())
        app.draw()
        key(app, pygame.K_TAB)
        key(app, pygame.K_F3)
        app.draw()
        assert app.expanded
    app.event(pygame.event.Event(pygame.VIDEORESIZE, w=1000, h=700))
    app.draw()
    assert app._viewport.width == 1000
    assert app._viewport.height < 700


def test_catchup_is_bounded_and_frame_schedules_match(app):
    app.start_race()
    assert app.advance(2, VehicleControl(throttle=1)) == 16
    assert app.accumulator < app.settings["physics_dt"]
    results = []
    for frequency in (30, 60, 144):
        app.start_race()
        for _ in range(frequency*4):
            app.advance(1/frequency, VehicleControl(throttle=.7))
        results.append(app.state)
    assert results[0] == results[1] == results[2]


def test_quit_event_exits_cleanly(app):
    app.event(pygame.event.Event(pygame.QUIT))
    assert not app.running


def test_slower_session_best_does_not_replace_saved_record(app):
    app.records.record(app.tracks[0], 30)
    app.start_race()
    app.simulation.countdown = 0
    track = app.tracks[0]
    x, y, heading = track.pose(track.total_length-.05)
    app.simulation.vehicle.x, app.simulation.vehicle.y, app.simulation.vehicle.heading = x, y, heading
    app.simulation.vehicle.longitudinal_velocity = 12
    app.simulation.projection = track.project(x,y)
    app.simulation.timing.next_checkpoint = len(track.checkpoints)
    app.simulation.timing.forward_distance = track.total_length-.05
    app.simulation.timing.finish_armed = True
    app.simulation.timing.elapsed = 40
    app.state = app.previous = app.simulation.snapshot()
    app.advance(1/120, VehicleControl())
    assert app.state.completed_laps == 1
    assert app.records.best(track) == 30
    assert "NEW PERSONAL BEST" not in app.toast


def test_keyboard_controls_cancel_opposed_steering_and_scale_sensitivity():
    controller = KeyboardController()
    keys = defaultdict(bool, {pygame.K_w: True, pygame.K_RIGHT: True})
    gentle = controller.action(keys, .4)
    strong = controller.action(keys, 1.0)
    assert gentle.throttle == strong.throttle == 1
    assert gentle.steering == .4
    assert strong.steering == 1
    keys[pygame.K_a], keys[pygame.K_DOWN] = True, True
    opposed = controller.action(keys, .8)
    assert opposed.steering == 0
    assert opposed.brake == 1
