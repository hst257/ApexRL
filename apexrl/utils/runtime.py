"""Locate read-only game resources separately from portable user saves."""
from pathlib import Path
import sys


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def default_save_directory(root: Path) -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent/"saves"
    return root/"saves"
