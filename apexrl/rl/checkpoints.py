"""Versioned policy contracts and configuration provenance for portable checkpoints."""
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

from apexrl.core import VehicleConfig, load_tracks
from .config import EnvConfig
from .observations import ACTION_VERSION, OBSERVATION_VERSION, observation_names


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def physics_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("controls", "vehicle", "physics", "geometry", "track", "collisions", "timing", "simulation"):
        digest.update((root/f"apexrl/core/{name}.py").read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".zip" else Path(str(path)+".zip")


def make_manifest(root: Path, config: dict, track_id: str, tracks=None) -> dict:
    tracks = tracks or load_tracks(root/"tracks")
    env_config = EnvConfig.from_dict(config["environment"])
    return {"format_version": 1, "observation_version": OBSERVATION_VERSION, "action_version": ACTION_VERSION,
            "observation_names": list(observation_names(env_config)), "action_names": ["steering", "signed_pedal"],
            "physics_fingerprint": physics_fingerprint(root), "track_versions": {track.id: track.version for track in tracks},
            "vehicle_config": asdict(VehicleConfig.load(root/"config/vehicle.json")),
            "training_config": config, "training_track": track_id,
            "python": platform.python_version(),
            "dependencies": {name: version(name) for name in ("gymnasium", "stable-baselines3", "torch", "numpy")}}


def save_checkpoint(model, path: str | Path, manifest: dict):
    path = model_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.zip")
    model.save(temporary)
    os.replace(temporary, path)
    metadata = dict(manifest, num_timesteps=int(model.num_timesteps), model_sha256=file_hash(path))
    write_json(path.with_suffix(".metadata.json"), metadata)
    return path


def read_checkpoint(path: str | Path, root: Path, tracks=None) -> tuple[Path, dict, EnvConfig]:
    path = model_path(path)
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("A model .zip and matching .metadata.json are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("action_version") != ACTION_VERSION or metadata.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("Unsupported checkpoint contract version")
    if metadata.get("model_sha256") != file_hash(path):
        raise ValueError("Model checksum does not match its metadata")
    if metadata.get("physics_fingerprint") != physics_fingerprint(root):
        raise ValueError("Checkpoint physics differs from this checkout")
    tracks = tracks or load_tracks(root/"tracks")
    if metadata.get("track_versions") != {track.id: track.version for track in tracks}:
        raise ValueError("Checkpoint track definitions differ from this checkout")
    if metadata.get("vehicle_config") != asdict(VehicleConfig.load(root/"config/vehicle.json")):
        raise ValueError("Checkpoint vehicle parameters differ from this checkout")
    config = EnvConfig.from_dict(metadata["training_config"]["environment"])
    if metadata.get("observation_names") != list(observation_names(config)):
        raise ValueError("Checkpoint observation names are incompatible")
    return path, metadata, config
