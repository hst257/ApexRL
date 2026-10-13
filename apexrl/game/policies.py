"""Discover local policy files without importing optional learning dependencies."""
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    path: Path
    label: str
    training_track: str
    timesteps: int
    source_timesteps: int = 0
    validation: tuple[tuple[str, int, float], ...] = ()
    demonstration_steps: int = 0
    best_laps: tuple[tuple[str, float], ...] = ()


def rl_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("gymnasium", "torch", "stable_baselines3"))


def discover_policies(root: Path) -> list[PolicyEntry]:
    """Read sidecars only; selecting a policy performs full compatibility checks."""
    entries = []
    for directory in (root/"runs", root/"models"):
        if not directory.exists():
            continue
        for sidecar in sorted(directory.rglob("*.metadata.json")):
            path = sidecar.with_name(sidecar.name.removesuffix(".metadata.json")+".zip")
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                steps = data["num_timesteps"]
                if not path.is_file() or not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
                    continue
                if data.get("format_version") != 1 or data.get("training_track") not in ("all", "rookie_ring", "apex_valley", "velocity_circuit"):
                    continue
                relative = path.relative_to(root)
                label = relative.with_suffix("").as_posix().removeprefix("runs/")
                promotion = data.get("promotion", {})
                summary = promotion.get("summary", {}) if isinstance(promotion, dict) else {}
                if not isinstance(summary, dict):
                    summary = {}
                validated, laps = [], []
                for track_id, result in summary.items():
                    count, success = result["episodes"], result["success_rate"]
                    if isinstance(count, int) and count > 0 and math.isfinite(success) and 0 <= success <= 1:
                        validated.append((track_id, count, float(success)))
                        lap = result.get("best_lap_seconds")
                        if isinstance(lap, (int, float)) and not isinstance(lap, bool) and math.isfinite(lap) and lap > 0:
                            laps.append((track_id, float(lap)))
                source_steps = data.get("source_timesteps", 0)
                if not isinstance(source_steps, int) or source_steps < 0:
                    source_steps = 0
                bootstrap = data.get("bootstrap", data.get("source_bootstrap", {}))
                demonstrations = bootstrap.get("demonstration_transitions", 0) if isinstance(bootstrap, dict) else 0
                if not isinstance(demonstrations, int) or isinstance(demonstrations, bool) or demonstrations < 0:
                    demonstrations = 0
                entries.append(PolicyEntry(path, label, data["training_track"], steps, source_steps, tuple(validated), demonstrations, tuple(laps)))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(entries, key=lambda entry: (0 if entry.path.parent == root/"models" else 1,
                                             -sum(rate >= .9 for _, _, rate in entry.validation),
                                             -entry.timesteps, entry.label))
