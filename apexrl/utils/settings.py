import json
import math
from pathlib import Path
from .persistence import JsonStore


class Settings:
    RESOLUTIONS = ((960, 540), (1280, 720), (1600, 900), (1920, 1080))

    def __init__(self, defaults: str | Path, store: JsonStore):
        self.store = store
        self.values = json.loads(Path(defaults).read_text(encoding="utf-8"))
        if not isinstance(self.values, dict):
            raise ValueError("game.json must contain an object")
        for name in ("physics_dt", "pixels_per_meter", "countdown_seconds", "shortcut_seconds", "recovery_seconds"):
            value = self.values.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid game configuration: {name}")
        if abs(self.values["physics_dt"]-1/120) > 1e-12:
            raise ValueError("Phase 1 requires physics_dt = 1/120")
        if not .5 <= self.values["pixels_per_meter"] <= 20:
            raise ValueError("pixels_per_meter must be in [0.5, 20]")
        if self.values["shortcut_seconds"] <= 0 or self.values["recovery_seconds"] <= self.values["shortcut_seconds"]:
            raise ValueError("recovery_seconds must follow shortcut_seconds")
        maximum = self.values.get("max_catchup_steps")
        if not isinstance(maximum, int) or not 1 <= maximum <= 64:
            raise ValueError("max_catchup_steps must be in [1, 64]")
        for name in ("volume", "steering_sensitivity", "camera_smoothing", "quality", "resolution", "fullscreen"):
            if name not in self.values or not self.set(name, self.values[name], persist=False):
                raise ValueError(f"Invalid game configuration: {name}")
        saved = store.read("settings")
        for name in ("volume", "steering_sensitivity", "camera_smoothing", "quality", "resolution", "fullscreen"):
            if name in saved:
                self.set(name, saved[name], persist=False)

    def __getitem__(self, name):
        return self.values[name]

    def set(self, name, value, persist=True):
        ranges = {"volume": (0.0, 1.0), "steering_sensitivity": (0.4, 1.0), "camera_smoothing": (2.0, 12.0)}
        if name in ranges:
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                return False
            value = max(ranges[name][0], min(ranges[name][1], float(value)))
        elif name == "resolution":
            if not isinstance(value, (list, tuple)) or tuple(value) not in self.RESOLUTIONS:
                return False
            value = list(value)
        elif name == "quality":
            if value not in ("low", "high"):
                return False
        elif name == "fullscreen":
            if not isinstance(value, bool):
                return False
        else:
            return False
        self.values[name] = value
        if persist:
            return self.store.write("settings", {key: self.values[key] for key in (*ranges, "resolution", "quality", "fullscreen")})
        return True
