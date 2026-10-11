import json
from apexrl.utils.persistence import JsonStore, LapRecords
from apexrl.utils.settings import Settings
from pathlib import Path
import pytest


def test_settings_and_records_roundtrip(tmp_path, tracks):
    store = JsonStore(tmp_path)
    path = Path(__file__).resolve().parents[1]/"config/game.json"
    settings = Settings(path, store)
    assert settings.set("volume", .7)
    assert Settings(path, JsonStore(tmp_path))["volume"] == .7
    records = LapRecords(store, "rules1")
    records.record(tracks[0], 80.25)
    records.record(tracks[0], 90)
    assert LapRecords(JsonStore(tmp_path), "rules1").best(tracks[0]) == 80.25
    assert LapRecords(JsonStore(tmp_path), "rules2").best(tracks[0]) is None


def test_corrupt_and_nonfinite_saves_fall_back(tmp_path, tracks):
    (tmp_path/"settings.json").write_text("broken", encoding="utf-8")
    store = JsonStore(tmp_path)
    assert store.read("settings") == {}
    assert store.warnings
    records = LapRecords(store, "rules")
    records.records[records.key(tracks[0])] = float("nan")
    assert records.best(tracks[0]) is None
    assert not records.record(tracks[0], -1)


def test_settings_validation_and_save_failure(tmp_path):
    path = Path(__file__).resolve().parents[1]/"config/game.json"
    blocker = tmp_path/"file"
    blocker.write_text("no directory", encoding="utf-8")
    store = JsonStore(blocker)
    settings = Settings(path, store)
    assert not settings.set("volume", float("inf"))
    assert not settings.set("resolution", [1, 1])
    assert not settings.set("volume", .6)
    assert settings["volume"] == .6
    assert store.warnings


def test_invalid_physics_configuration_fails_early(tmp_path):
    defaults = json.loads((Path(__file__).resolve().parents[1]/"config/game.json").read_text())
    defaults["physics_dt"] = 0
    path = tmp_path/"game.json"
    path.write_text(json.dumps(defaults), encoding="utf-8")
    with pytest.raises(ValueError, match="physics_dt"):
        Settings(path, JsonStore(tmp_path/"saves"))
