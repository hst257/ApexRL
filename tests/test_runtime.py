from pathlib import Path
import sys
from apexrl.utils.runtime import default_save_directory, resource_root


def test_source_paths_do_not_depend_on_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = resource_root()
    assert (root/"config/vehicle.json").is_file()
    assert default_save_directory(root) == root/"saves"


def test_frozen_resources_and_saves_are_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path/"_internal"), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path/"ApexRL.exe"))
    assert resource_root() == tmp_path/"_internal"
    assert default_save_directory(resource_root()) == tmp_path/"saves"
