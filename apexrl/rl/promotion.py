"""Publish a local policy only after independent seeded lap-completion checks."""
import hashlib
import math
from pathlib import Path
import re

from .checkpoints import model_path, read_checkpoint, write_json


def qualifies(report: dict, minimum_success: float) -> bool:
    if not math.isfinite(minimum_success) or not 0 < minimum_success <= 1:
        raise ValueError("minimum_success must be in (0, 1]")
    summaries = report.get("summary", {})
    return bool(summaries) and all(value["success_rate"] >= minimum_success and
                                   value["best_lap_seconds"] is not None for value in summaries.values())


def validate_name(name: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) or name.lower() in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1,10)), *(f"lpt{i}" for i in range(1,10))}:
        raise ValueError("Use a nonreserved lowercase model name containing letters, digits, underscores, or hyphens")


def publish_policy(source: Path, root: Path, name: str, report: dict, minimum_success: float) -> Path:
    validate_name(name)
    if not qualifies(report, minimum_success):
        raise ValueError("Policy did not meet the required held-out valid-lap success rate")
    checked, metadata, _ = read_checkpoint(source, root)
    if report.get("model_sha256") != metadata["model_sha256"]:
        raise ValueError("Evaluation does not match the checkpoint being published")
    destination = model_path(root/"models"/name)
    sidecar = destination.with_suffix(".metadata.json")
    report_path = destination.with_suffix(".evaluation.json")
    if any(path.exists() for path in (destination, sidecar, report_path)):
        raise ValueError("Published model name already exists; use a new name")
    contents = checked.read_bytes()
    if hashlib.sha256(contents).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Checkpoint changed during publication")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing models even if another publisher races us.
    with destination.open("xb") as handle:
        handle.write(contents)
    metadata["promotion"] = {"minimum_success": minimum_success, "evaluation_seed": report["seed"],
                             "summary": report["summary"], "source": str(checked.resolve())}
    write_json(sidecar, metadata)
    write_json(report_path, report)
    return destination
