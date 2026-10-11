import json
import math
import os
from pathlib import Path


class JsonStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.warnings: list[str] = []

    def read(self, name: str) -> dict:
        path = self.directory/f"{name}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("data"), dict):
                raise ValueError("Unknown save format")
            return data["data"]
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError):
            self.warnings.append(f"Could not read {name}; using defaults.")
            return {}

    def write(self, name: str, data: dict) -> bool:
        temporary = self.directory/f"{name}.tmp"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            text = json.dumps({"version": 1, "data": data}, indent=2, allow_nan=False)
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, self.directory/f"{name}.json")
            return True
        except (OSError, ValueError, TypeError):
            self.warnings.append(f"Could not save {name}; changes remain in memory.")
            return False


class LapRecords:
    def __init__(self, store: JsonStore, rules_version: str):
        self.store, self.rules_version = store, rules_version
        self.records = store.read("records")

    def key(self, track) -> str:
        return f"{track.id}:{track.version}:{self.rules_version}"

    def best(self, track) -> float | None:
        value = self.records.get(self.key(track))
        return float(value) if isinstance(value, (float, int)) and math.isfinite(value) and value > 0 else None

    def record(self, track, time: float) -> bool:
        if not math.isfinite(time) or time <= 0:
            return False
        old = self.best(track)
        if old is None or time < old:
            self.records[self.key(track)] = time
            return self.store.write("records", self.records)
        return True
