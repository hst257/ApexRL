"""Optional spectator rendering. This module is never imported by headless runs."""
import numpy as np
import pygame
from apexrl.game.camera import Camera
from apexrl.game.renderer import Renderer, INK, LIME
from apexrl.utils.persistence import JsonStore
from apexrl.utils.settings import Settings


class PolicyViewer:
    def __init__(self, mode, root, vehicle_config):
        self.mode = mode
        pygame.font.init()
        self.display = None
        if mode == "human":
            pygame.display.init()
            self.display = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
            pygame.display.set_caption("APEX RL — AI Driver")
        self.canvas = pygame.Surface((1280, 720))
        self.renderer = Renderer()
        self.renderer.vehicle_config = vehicle_config
        # Read defaults, never player preferences or personal-best records.
        self.settings = Settings(root/"config/game.json", JsonStore(root/"artifacts/viewer-defaults"))
        self.camera = Camera()
        self.clock = pygame.time.Clock()
        self.expanded = self.debug = False
        self.close_requested = False

    def reset(self, vehicle):
        self.camera.reset(vehicle)
        self.renderer.clear_effects()

    def draw(self, simulation, info):
        dt = info.get("physics_ticks", 4)/120
        dt = max(1/120, dt)
        if self.mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.close_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB:
                        self.expanded = not self.expanded
                    elif event.key == pygame.K_F3:
                        self.debug = not self.debug
                elif event.type == pygame.VIDEORESIZE:
                    self.display = pygame.display.set_mode((max(640, event.w), max(360, event.h)), pygame.RESIZABLE)
        state = simulation.snapshot()
        self.camera.update(state.vehicle, dt, self.settings["camera_smoothing"], self.settings["pixels_per_meter"])
        self.renderer.effects(state.vehicle, state, dt, self.settings["quality"])
        self.renderer.race(self.canvas, simulation.track, state.vehicle, state, self.camera,
                           self.settings, state.best_lap, self.expanded, self.debug,
                           (self.clock.get_fps(), 120))
        if not self.expanded:
            pygame.draw.rect(self.canvas, INK, (370, 111, 580, 36))
            self.renderer.type.text(self.canvas, f"AI DRIVER  /  PROGRESS {info.get('progress_fraction', 0)*100:05.1f}%  /  {info.get('episode_seconds', 0):.1f} S", (660, 121), 16, LIME, True, "center")
            pygame.draw.rect(self.canvas, INK, (600, 660, 680, 60))
            self.renderer.type.text(self.canvas, "TAB  CIRCUIT MAP     F3  TELEMETRY     ESC  EXIT", (1252, 679), 15, LIME, align="right")
        if self.mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.canvas), (1, 0, 2)).copy()
        width, height = self.display.get_size()
        scale = min(width/1280, height/720)
        size = (round(1280*scale), round(720*scale))
        self.display.fill(INK)
        self.display.blit(pygame.transform.scale(self.canvas, size), ((width-size[0])//2, (height-size[1])//2))
        pygame.display.flip()
        self.clock.tick(round(1/dt))
        return None

    def close(self):
        if self.display is not None:
            pygame.display.quit()
