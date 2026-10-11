from pathlib import Path
import pytest
from apexrl.core import load_tracks, VehicleConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tracks():
    return load_tracks(ROOT/"tracks")


@pytest.fixture(scope="session")
def vehicle_config():
    return VehicleConfig.load(ROOT/"config/vehicle.json")
