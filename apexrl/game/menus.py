from dataclasses import dataclass
import pygame
from .renderer import INK, PANEL, WHITE, MUTED, LIME, RED, format_time


@dataclass(slots=True)
class Button:
    rect: pygame.Rect
    action: str
    label: str


class Menus:
    def __init__(self, renderer):
        self.renderer = renderer
        self.type = renderer.type
        self.buttons = []
        self.selected = 0

    def clear(self):
        self.buttons = []
        self.selected = 0

    def button(self, surface, rect, label, action, subtitle=None):
        rect = pygame.Rect(rect)
        index = len(self.buttons)
        active = index == self.selected
        self.buttons.append(Button(rect, action, label))
        pygame.draw.rect(surface, LIME if active else PANEL, rect, border_radius=4)
        self.type.text(surface, label, (rect.x+18, rect.y+12), 21, INK if active else WHITE, True)
        arrow_color = INK if active else MUTED
        pygame.draw.lines(surface, arrow_color, False, [(rect.right-27, rect.centery-5), (rect.right-21, rect.centery), (rect.right-27, rect.centery+5)], 2)
        if subtitle:
            self.type.text(surface, subtitle, (rect.x+18, rect.y+40), 14, MUTED)

    def background(self, surface, title, eyebrow):
        surface.fill(INK)
        for x in range(-720, 1280, 72):
            pygame.draw.line(surface, (20, 35, 36), (x, 720), (x+720, 0), 1)
        pygame.draw.rect(surface, LIME, (64, 38, 38, 4))
        self.type.text(surface, eyebrow.upper(), (118, 29), 15, MUTED, True)
        self.type.text(surface, title, (64, 72), 48, WHITE, True)
        self.type.text(surface, "APEX RL  /  ORIGINAL CIRCUITS. YOUR RACING LINE.", (64, 681), 13, MUTED)
        self.type.text(surface, "UP / DOWN  NAVIGATE     ENTER  SELECT     ESC  BACK", (1216, 681), 13, MUTED, align="right")

    def draw(self, surface, screen, tracks, current, records, settings, audio_available):
        self.buttons = []
        if screen == "main":
            self.background(surface, "", "TIME TRIAL / SINGLE PLAYER")
            self.type.text(surface, "APEX", (58, 64), 116, WHITE, True)
            self.type.text(surface, "RL", (350, 95), 74, LIME, True)
            self.type.text(surface, "CHASE YOUR PERFECT LAP.", (65, 210), 25, WHITE, True)
            self.type.text(surface, "Three circuits. One car. Every tenth matters.", (65, 248), 17, MUTED)
            for i, (label, action) in enumerate((("Play", "play"), ("Track Selection", "tracks"), ("Controls", "controls"), ("Settings", "settings"), ("Quit", "quit"))):
                self.button(surface, (64, 315+i*61, 360, 49), label.upper(), action)
            self.type.text(surface, "ON THE GRID", (510, 153), 14, LIME, True)
            self.renderer.minimap(surface, tracks[current], (508, 191, 704, 345))
            track = tracks[current]
            self.type.text(surface, track.name.upper(), (510, 558), 30, WHITE, True)
            self.type.text(surface, f"{track.difficulty.upper()}  /  {track.total_length/1000:.2f} KM  /  PERSONAL BEST {format_time(records.best(track))}", (510, 605), 15, MUTED)
        elif screen == "tracks":
            self.background(surface, "Choose your circuit.", "THE TRACK COLLECTION")
            for i, track in enumerate(tracks):
                x = 64+i*388
                active = self.selected == i
                pygame.draw.rect(surface, PANEL, (x, 177, 360, 414), border_radius=6)
                pygame.draw.rect(surface, LIME if active else (47, 69, 64), (x, 177, 360, 4))
                self.type.text(surface, f"0{i+1} / {track.difficulty.upper()}", (x+20, 199), 15, LIME)
                self.renderer.minimap(surface, track, (x+15, 237, 330, 160))
                self.type.text(surface, track.name, (x+20, 412), 28, WHITE, True)
                self.type.text(surface, f"{track.total_length/1000:.2f} KM   •   {float(track.widths.min()):.0f}–{float(track.widths.max()):.0f} M WIDE", (x+20, 455), 14, MUTED)
                self.type.text(surface, f"BEST  {format_time(records.best(track))}", (x+20, 485), 17, WHITE, mono=True)
                self.button(surface, (x+20, 528, 320, 45), "DRIVE CIRCUIT", f"track:{i}")
            self.button(surface, (64, 612, 180, 44), "BACK", "back")
        elif screen == "controls":
            self.background(surface, "Make every input count.", "DRIVER HANDBOOK")
            rows = [("W / UP", "Accelerate"), ("S / DOWN", "Brake / stop"), ("A / LEFT", "Steer left"), ("D / RIGHT", "Steer right"),
                    ("R", "Recover car / invalidate lap"), ("ESC", "Pause / resume"), ("TAB", "Expand circuit minimap"), ("F3", "Developer telemetry")]
            for i, (key, label) in enumerate(rows):
                x, y = 64+(i//4)*585, 185+(i%4)*75
                pygame.draw.rect(surface, PANEL, (x, y, 140, 53), border_radius=4)
                self.type.text(surface, key, (x+16, y+15), 18, LIME, True, mono=True)
                self.type.text(surface, label, (x+160, y+15), 20, WHITE)
            self.type.text(surface, "Brake before the corner. Ease off the throttle to recover a slide.", (64, 532), 21, WHITE)
            self.type.text(surface, "Grass reduces grip. Follow checkpoint order; recovered and shortcut laps cannot set records.", (64, 574), 17, MUTED)
            self.button(surface, (64, 614, 220, 44), "BACK", "back")
        elif screen == "settings":
            self.background(surface, "Tune your drive.", "SETTINGS / SAVED LOCALLY")
            labels = [("volume", "MASTER VOLUME", f"{settings['volume']*100:.0f}%"),
                      ("steering_sensitivity", "STEERING SENSITIVITY", f"{settings['steering_sensitivity']:.1f}×"),
                      ("camera_smoothing", "CAMERA RESPONSE", f"{settings['camera_smoothing']:.0f}"),
                      ("quality", "GRAPHICS DETAIL", settings["quality"].upper()),
                      ("resolution", "DISPLAY RESOLUTION", " × ".join(map(str, settings["resolution"]))),
                      ("fullscreen", "WINDOW MODE", "FULLSCREEN" if settings["fullscreen"] else "WINDOWED")]
            for i, (name, label, value) in enumerate(labels):
                self.button(surface, (64, 171+i*67, 940, 51), label, f"setting:{name}")
                self.type.text(surface, value, (947, 185+i*67), 21, INK if self.selected == i else LIME, True, "right", True)
            self.type.text(surface, "LEFT / RIGHT  ADJUST VALUES   /   ENTER OR CLICK TO CYCLE", (64, 594), 15, MUTED)
            if not audio_available:
                self.type.text(surface, "Audio device unavailable; gameplay remains available.", (64, 625), 15, RED)
            self.button(surface, (1052, 589, 164, 50), "BACK", "back")
        elif screen == "pause":
            veil = pygame.Surface((1280, 720), pygame.SRCALPHA)
            veil.fill((*INK, 215))
            surface.blit(veil, (0, 0))
            self.type.text(surface, "TAKE A BREATHER.", (640, 108), 43, WHITE, True, "center")
            self.type.text(surface, "THE CLOCK IS PAUSED.", (640, 166), 15, LIME, True, "center")
            for i, (label, action) in enumerate((("Resume", "resume"), ("Restart", "restart"), ("Track Selection", "tracks"), ("Settings", "settings"), ("Main Menu", "main"))):
                self.button(surface, (440, 230+i*69, 400, 52), label.upper(), action)

    def navigate(self, direction):
        if self.buttons:
            self.selected = (self.selected+direction)%len(self.buttons)

    def hover(self, position):
        for index, button in enumerate(self.buttons):
            if button.rect.collidepoint(position):
                self.selected = index
                return button.action
        return None

    def selected_action(self):
        return self.buttons[self.selected].action if self.buttons else None
