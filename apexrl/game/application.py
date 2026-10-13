from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import pickle
from pathlib import Path
import zipfile
import pygame

from apexrl.core import RacingSimulation, VehicleConfig, VehicleControl, load_tracks
from apexrl.core.simulation import SimulationConfig
from apexrl.utils.persistence import JsonStore, LapRecords
from apexrl.utils.settings import Settings
from apexrl.utils.runtime import default_save_directory
from .identity import app_icon
from .audio import Audio
from .camera import Camera
from .input_handler import KeyboardController
from .menus import Menus
from .renderer import Renderer, INK, RED, LIME, MUTED, format_time
from .policies import discover_policies, rl_available


class Application:
    WIDTH, HEIGHT = 1280, 720

    def __init__(self, root: str | Path, save_directory: str | Path | None = None):
        self.root = Path(root)
        self.store = JsonStore(save_directory or default_save_directory(self.root))
        self.settings = Settings(self.root/"config/game.json", self.store)
        self.vehicle_config = VehicleConfig.load(self.root/"config/vehicle.json")
        self.tracks = load_tracks(self.root/"tracks")
        rule_definition = {"vehicle": json.loads((self.root/"config/vehicle.json").read_text()),
                           "shortcut_seconds": self.settings["shortcut_seconds"],
                           "recovery_seconds": self.settings["recovery_seconds"],
                           "physics": "phase1-bicycle-v1"}
        rules = hashlib.sha256(json.dumps(rule_definition, sort_keys=True).encode()).hexdigest()[:12]
        self.records = LapRecords(self.store, rules)
        pygame.display.init()
        pygame.font.init()
        self.display = None
        self.apply_display()
        pygame.display.set_caption("APEX RL — Time Trial")
        pygame.display.set_icon(app_icon())
        self.canvas = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.renderer = Renderer()
        self.renderer.vehicle_config = self.vehicle_config
        self.menus = Menus(self.renderer)
        self.camera = Camera()
        self.controller = KeyboardController()
        self.audio = Audio()
        self.clock = pygame.time.Clock()
        self.current_track = 0
        self.ai_session = None
        self.ai_models = []
        self.ai_show_training = False
        self.ai_loader = None
        self.ai_future = None
        self.ai_model_index = 0
        self.ai_error = ""
        self.simulation = None
        self.previous = self.state = None
        self.screen = "main"
        self.focused = True
        self.settings_origin = "main"
        self.tracks_origin = "main"
        self.running = True
        self.accumulator = 0.0
        self.expanded = self.debug = False
        self.fps = self.steps_per_second = 0.0
        self.metric_elapsed = self.metric_steps = 0
        self.last_count = 4
        self.toast = ""
        self.toast_seconds = 0.0
        self._viewport = pygame.Rect(0, 0, *self.settings["resolution"])
        self.draw()

    def apply_display(self):
        flags = pygame.FULLSCREEN if self.settings["fullscreen"] else pygame.RESIZABLE
        try:
            self.display = pygame.display.set_mode(tuple(self.settings["resolution"]), flags)
        except pygame.error:
            self.settings.set("fullscreen", False)
            self.display = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
            self.store.warnings.append("Requested display mode unavailable; using a window.")

    def transition(self, screen):
        if self.screen == "ai_loading" and screen != "ai_loading":
            self.cancel_ai_load()
        if self.screen in ("ai_race", "ai_pause") and screen not in ("ai_race", "ai_pause"):
            self.toast_seconds = 0
        self.screen = screen
        self.menus.clear()
        self.accumulator = 0.0

    def start_race(self, index=None):
        if self.ai_session:
            self.ai_session.close()
            self.ai_session = None
        if index is not None:
            self.current_track = index
        config = SimulationConfig(self.settings["countdown_seconds"], self.settings["shortcut_seconds"], self.settings["recovery_seconds"])
        self.simulation = RacingSimulation(self.tracks[self.current_track], self.vehicle_config, simulation_config=config)
        self.state = self.previous = self.simulation.snapshot()
        self.camera.reset(self.state.vehicle)
        self.renderer.clear_effects()
        self.expanded = False
        self.last_count = 4
        self.transition("race")

    def refresh_policies(self) -> None:
        selected = self.ai_models[self.ai_model_index].path if self.ai_models else None
        entries = discover_policies(self.root)
        published = [entry for entry in entries if entry.path.is_relative_to(self.root/"models")]
        self.ai_models = entries if self.ai_show_training or not published else published
        self.ai_model_index = next((i for i, entry in enumerate(self.ai_models) if entry.path == selected), 0)
        self.ai_error = "" if rl_available() else "Install requirements-rl.txt in this Python environment to watch AI."

    def install_ai(self, candidate) -> None:
        if self.ai_session:
            self.ai_session.close()
        self.ai_session = candidate
        self.camera.reset(candidate.simulation.snapshot().vehicle)
        self.renderer.clear_effects()
        self.expanded = False
        self.ai_error = ""
        self.toast_seconds = 0
        self.transition("ai_race" if self.focused else "ai_pause")

    def request_ai(self) -> None:
        if not rl_available() or not self.ai_models:
            self.start_ai()
            return
        if self.ai_future is not None:
            return
        from .ai_session import AISession
        if self.ai_loader is None:
            self.ai_loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apexrl-policy")
        self.ai_future = self.ai_loader.submit(AISession, self.root,
                                             self.ai_models[self.ai_model_index].path,
                                             self.tracks[self.current_track].id, self.tracks)
        self.transition("ai_loading")

    def cancel_ai_load(self) -> None:
        future, self.ai_future = self.ai_future, None
        if future is None:
            return
        if not future.cancel():
            def close_unused(completed):
                if not completed.cancelled() and completed.exception() is None:
                    completed.result().close()
            future.add_done_callback(close_unused)

    def poll_ai_load(self) -> None:
        if self.ai_future is None or not self.ai_future.done():
            return
        future, self.ai_future = self.ai_future, None
        try:
            candidate = future.result()
        except Exception as error:
            self.ai_error = f"Could not load driver: {error}"
            self.transition("ai")
        else:
            self.install_ai(candidate)

    def start_ai(self) -> bool:
        if not rl_available():
            self.ai_error = "Install requirements-rl.txt in this Python environment to watch AI."
            return False
        if not self.ai_models:
            self.ai_error = "No local models found. Train a policy, then refresh."
            return False
        try:
            from .ai_session import AISession
            candidate = AISession(self.root, self.ai_models[self.ai_model_index].path,
                                  self.tracks[self.current_track].id, self.tracks)
        except (OSError, ValueError, ImportError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
            self.ai_error = f"Could not load policy: {error}"
            return False
        self.install_ai(candidate)
        return True

    def adjust(self, name, direction):
        value = self.settings[name]
        if name in ("volume", "steering_sensitivity", "camera_smoothing"):
            increment = {"volume": .1, "steering_sensitivity": .1, "camera_smoothing": 1}[name]
            value = round(value+increment*direction, 2)
        elif name == "resolution":
            options = Settings.RESOLUTIONS
            value = options[(options.index(tuple(value))+direction)%len(options)]
        elif name == "fullscreen":
            value = not value
        else:
            value = "low" if value == "high" else "high"
        self.settings.set(name, value)
        if name in ("resolution", "fullscreen"):
            self.apply_display()

    def activate(self, action):
        if action is None:
            return
        self.audio.effect("click", self.settings["volume"])
        if action == "quit":
            self.running = False
        elif action == "play" or action == "restart":
            self.start_race()
        elif action.startswith("track:"):
            self.start_race(int(action.split(":")[1]))
        elif action == "resume":
            self.transition("race")
        elif action == "ai":
            self.refresh_policies()
            self.transition("ai")
        elif action.startswith("ai:"):
            operation = action.split(":")[1]
            if operation == "watch":
                self.request_ai()
            elif operation in ("next", "previous") and self.ai_models:
                self.ai_model_index = (self.ai_model_index+(1 if operation == "next" else -1)) % len(self.ai_models)
                self.ai_error = ""
            elif operation == "track":
                self.current_track = (self.current_track+1) % len(self.tracks)
            elif operation == "refresh":
                self.refresh_policies()
            elif operation == "library":
                self.ai_show_training = not self.ai_show_training
                self.refresh_policies()
            elif operation == "cancel":
                self.transition("ai")
            elif operation == "resume":
                self.transition("ai_race")
            elif operation == "restart" and self.ai_session:
                self.ai_session.reset()
                self.camera.reset(self.ai_session.simulation.snapshot().vehicle)
                self.renderer.clear_effects()
                self.toast_seconds = 0
                self.transition("ai_race")
        elif action == "tracks":
            self.tracks_origin = "pause" if self.simulation and self.screen == "pause" else "main"
            self.transition("tracks")
        elif action == "settings":
            self.settings_origin = self.screen
            self.transition("settings")
        elif action == "back":
            destination = self.settings_origin if self.screen == "settings" else self.tracks_origin if self.screen == "tracks" else "main"
            self.transition(destination)
        elif action.startswith("setting:"):
            self.adjust(action.split(":")[1], 1)
        else:
            self.transition(action)

    def event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.focused = False
            if self.screen in ("race", "ai_race"):
                self.transition("pause" if self.screen == "race" else "ai_pause")
        elif event.type == pygame.WINDOWFOCUSGAINED:
            self.focused = True
        elif event.type == pygame.VIDEORESIZE and not self.settings["fullscreen"]:
            self.display = pygame.display.set_mode((max(640, event.w), max(360, event.h)), pygame.RESIZABLE)
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_ESCAPE:
                if self.screen == "race":
                    self.transition("pause")
                elif self.screen == "pause":
                    self.transition("race")
                elif self.screen == "ai_race":
                    self.transition("ai_pause")
                elif self.screen == "ai_pause":
                    self.transition("ai_race")
                elif self.screen == "ai_loading":
                    self.transition("ai")
                elif self.screen != "main":
                    self.activate("back")
            elif self.screen == "ai_race":
                if key == pygame.K_r:
                    self.activate("ai:restart")
                elif key == pygame.K_TAB:
                    self.expanded = not self.expanded
                elif key == pygame.K_F3:
                    self.debug = not self.debug
            elif self.screen == "race":
                if key == pygame.K_r:
                    result = self.simulation.recover_car()
                    self.state = self.previous = result.state
                    self.camera.reset(self.state.vehicle)
                    self.renderer.clear_effects()
                    self.accumulator = 0
                elif key == pygame.K_TAB:
                    self.expanded = not self.expanded
                elif key == pygame.K_F3:
                    self.debug = not self.debug
            elif key in (pygame.K_UP, pygame.K_DOWN):
                self.menus.navigate(-1 if key == pygame.K_UP else 1)
            elif key in (pygame.K_LEFT, pygame.K_RIGHT) and self.screen == "settings":
                action = self.menus.selected_action()
                if action and action.startswith("setting:"):
                    self.adjust(action.split(":")[1], -1 if key == pygame.K_LEFT else 1)
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self.activate(self.menus.selected_action())
        elif event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN) and self.screen not in ("race", "ai_race"):
            x, y = event.pos
            pos = ((x-self._viewport.x)*1280/self._viewport.width, (y-self._viewport.y)*720/self._viewport.height)
            action = self.menus.hover(pos)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.activate(action)

    def advance(self, elapsed: float, action: VehicleControl | None = None):
        self.poll_ai_load()
        elapsed = max(0.0, elapsed)
        self.toast_seconds = max(0, self.toast_seconds-elapsed)
        steps = 0
        if self.screen == "ai_race" and self.ai_session:
            session = self.ai_session
            self.accumulator += min(elapsed, .25)
            while not session.finished and self.accumulator+1e-12 >= session.step_seconds and steps < self.settings["max_catchup_steps"]:
                info = session.step()
                steps += info["physics_ticks"]
                self.accumulator -= session.step_seconds
                if session.finished:
                    self.toast = ("AI VALID LAP / "+format_time(info["lap_time"])) if info["is_success"] else "AI EPISODE / "+info["termination_reason"].replace("_", " ").upper()
                    self.toast_seconds = 3600
            if session.finished or self.accumulator >= session.step_seconds:
                self.accumulator = 0
            state = session.simulation.snapshot()
            self.camera.update(state.vehicle, elapsed, self.settings["camera_smoothing"], self.settings["pixels_per_meter"])
            self.renderer.effects(state.vehicle, state, min(elapsed, .1), self.settings["quality"])
        elif self.screen == "race":
            if action is None:
                action = self.controller.action(pygame.key.get_pressed(), self.settings["steering_sensitivity"])
            self.accumulator += min(elapsed, .25)
            dt = self.settings["physics_dt"]
            while self.accumulator+1e-12 >= dt and steps < self.settings["max_catchup_steps"]:
                self.previous = self.state
                result = self.simulation.step(action, dt)
                self.state = result.state
                for event in result.events:
                    if event.kind == "best_lap":
                        prior_best = self.records.best(self.tracks[self.current_track])
                        self.records.record(self.tracks[self.current_track], event.time)
                        if prior_best is None or event.time < prior_best:
                            self.toast, self.toast_seconds = "NEW PERSONAL BEST", 4.0
                            self.audio.effect("lap", self.settings["volume"])
                    elif event.kind == "lap_completed" and self.toast_seconds <= 0:
                        self.toast, self.toast_seconds = "LAP COMPLETE / "+format_time(event.time), 4.0
                        self.audio.effect("lap", self.settings["volume"])
                    elif event.kind == "collision" and not self.previous.collision:
                        self.audio.effect("collision", self.settings["volume"])
                    elif event.kind == "recovered":
                        self.previous = self.state
                        self.camera.reset(self.state.vehicle)
                        self.renderer.clear_effects()
                self.accumulator -= dt
                steps += 1
            if self.accumulator >= dt:
                self.accumulator %= dt
            count = math.ceil(self.state.countdown)
            if count != self.last_count:
                self.audio.effect("count" if count else "go", self.settings["volume"])
                self.last_count = count
            self.camera.update(self.interpolated_vehicle(), elapsed, self.settings["camera_smoothing"], self.settings["pixels_per_meter"])
            self.renderer.effects(self.state.vehicle, self.state, min(elapsed, .1), self.settings["quality"])
        else:
            self.accumulator = 0.0
        sound_vehicle = self.ai_session.simulation.vehicle if self.screen == "ai_race" and self.ai_session else self.state.vehicle if self.state else None
        sound_active = ((self.screen == "ai_race" and self.ai_session and not self.ai_session.finished)
                        or (self.screen == "race" and self.state.countdown == 0))
        self.audio.update(sound_vehicle, self.settings["volume"], bool(sound_active))
        self.metric_elapsed += elapsed
        self.metric_steps += steps
        if self.metric_elapsed >= .5:
            self.steps_per_second = self.metric_steps/self.metric_elapsed
            self.metric_elapsed = self.metric_steps = 0
        return steps

    def interpolated_vehicle(self):
        a, b = self.previous.vehicle, self.state.vehicle
        fraction = min(1, self.accumulator/self.settings["physics_dt"])
        angle = (b.heading-a.heading+math.pi) % (2*math.pi)-math.pi
        return replace(b, x=a.x+(b.x-a.x)*fraction, y=a.y+(b.y-a.y)*fraction, heading=a.heading+angle*fraction)

    def draw(self):
        if self.screen in ("ai_race", "ai_pause") and self.ai_session:
            session = self.ai_session
            state = session.simulation.snapshot()
            self.renderer.race(self.canvas, session.simulation.track, state.vehicle, state, self.camera, self.settings,
                               state.best_lap, self.expanded, self.debug, (self.fps, self.steps_per_second))
            pygame.draw.rect(self.canvas, INK, (535, 18, 180, 24))
            self.renderer.type.text(self.canvas, "AI SESSION BEST", (535, 22), 14, MUTED)
            if not self.expanded:
                pygame.draw.rect(self.canvas, INK, (370, 111, 580, 36))
                self.renderer.type.text(self.canvas, f"AI DRIVER / LAP PROGRESS {session.info['max_progress_fraction']:.1%}", (660, 121), 16, LIME, True, "center")
                pygame.draw.rect(self.canvas, INK, (600, 660, 680, 60))
                self.renderer.type.text(self.canvas, "R  RESTART     TAB  MAP     F3  TELEMETRY     ESC  PAUSE", (1252, 679), 15, LIME, align="right")
        if self.screen in ("race", "pause") and self.state:
            track = self.tracks[self.current_track]
            self.renderer.race(self.canvas, track, self.interpolated_vehicle(), self.state, self.camera, self.settings,
                               self.records.best(track), self.expanded, self.debug, (self.fps, self.steps_per_second))
        if self.screen not in ("race", "ai_race"):
            selected = self.ai_models[self.ai_model_index] if self.ai_models else None
            details = {"label": selected.label if selected else "No checkpoint found.", "error": self.ai_error,
                       "summary": f"{self.ai_model_index+1} / {len(self.ai_models)} POLICIES  /  {selected.timesteps:,} TRAINING STEPS  /  {selected.training_track.upper()}" if selected else "Train a policy with train.py to get started."}
            if selected and selected.source_timesteps:
                details["summary"] = f"{selected.timesteps:,} NEW STEPS  /  {selected.source_timesteps:,} SOURCE STEPS  /  {selected.training_track.upper()}"
            if selected and selected.demonstration_steps:
                details["summary"] = f"{selected.timesteps:,} PPO STEPS  /  {selected.demonstration_steps:,} DEMONSTRATIONS  /  {selected.training_track.upper()}"
            if selected and selected.validation:
                current = self.tracks[self.current_track].id
                result = next((value for value in selected.validation if value[0] == current), None)
                details["validation"] = f"HELD-OUT VALID LAPS {round(result[1]*result[2])}/{result[1]} ON THIS CIRCUIT" if result else "THIS POLICY HAS NO PUBLICATION RESULT FOR THE SELECTED CIRCUIT"
                reliable = sum(rate >= .9 for _, _, rate in selected.validation)
                details["summary"] = f"VALIDATED ON {reliable} OF 3 CIRCUITS  /  {self.ai_model_index+1} OF {len(self.ai_models)} DRIVERS"
            if selected:
                details["label"] = selected.path.stem.replace("-", " ").replace("_", " ").title()
                lap = next((value for identifier, value in selected.best_laps if identifier == self.tracks[self.current_track].id), None)
                details["best_lap"] = format_time(lap)
            details["library"] = "Published drivers" if not self.ai_show_training else "All checkpoints"
            self.menus.draw(self.canvas, self.screen, self.tracks, self.current_track, self.records, self.settings, self.audio.available, details)
        if self.toast_seconds > 0 and self.screen != "ai_pause":
            y = 170 if self.screen in ("ai_race", "ai_pause") else 118
            if y == 170:
                pygame.draw.rect(self.canvas, INK, (320, 159, 640, 42))
            self.renderer.type.text(self.canvas, self.toast, (640, y), 20, (205, 247, 92), True, "center")
        if self.store.warnings:
            self.renderer.type.text(self.canvas, self.store.warnings[-1], (640, 650), 15, RED, align="center")
        width, height = self.display.get_size()
        scale = min(width/1280, height/720)
        size = (max(1, round(1280*scale)), max(1, round(720*scale)))
        self._viewport = pygame.Rect((width-size[0])//2, (height-size[1])//2, *size)
        self.display.fill(INK)
        self.display.blit(pygame.transform.scale(self.canvas, size), self._viewport)
        pygame.display.flip()

    def run(self, frames: int | None = None, screenshot: str | None = None):
        count = 0
        try:
            while self.running and (frames is None or count < frames):
                elapsed = self.clock.tick(60)/1000
                self.fps = self.clock.get_fps()
                for event in pygame.event.get():
                    self.event(event)
                if not self.running:
                    break
                self.advance(elapsed)
                self.draw()
                count += 1
            if screenshot:
                path = Path(screenshot)
                path.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(self.canvas, str(path))
        finally:
            self.close()

    def close(self):
        self.cancel_ai_load()
        if self.ai_loader:
            self.ai_loader.shutdown(wait=False, cancel_futures=True)
        if self.ai_session:
            self.ai_session.close()
        self.audio.close()
        pygame.quit()
