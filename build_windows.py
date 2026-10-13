"""Build the Windows 1.0 game with validated drivers and a matching source archive."""
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def collect_notices(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = ("pygame", "numpy", "torch", "gymnasium", "stable-baselines3", "cloudpickle", "sympy", "mpmath",
             "filelock", "fsspec", "networkx", "jinja2", "MarkupSafe", "typing_extensions", "packaging", "setuptools", "pyinstaller")
    versions = {}
    for name in names:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        versions[name] = distribution.version
        for relative in distribution.files or ():
            if not any(word in relative.name.lower() for word in ("license", "copying", "notice")):
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file() or source.suffix in (".py", ".pyc", ".pyd", ".dll"):
                continue
            target = destination/name/Path(*(part for part in relative.parts if part not in (".", "..")))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    python_license = Path(sys.base_prefix)/"LICENSE.txt"
    if python_license.exists():
        shutil.copyfile(python_license, destination/"PYTHON-LICENSE.txt")
    (destination/"DEPENDENCIES.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


def zip_directory(directory: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(directory.rglob("*")):
            if file.is_file() and "saves" not in file.relative_to(directory).parts:
                archive.write(file, directory.name+"/"+file.relative_to(directory).as_posix())


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("Build the Windows distribution on Windows")
    from apexrl import __version__
    from apexrl.rl.checkpoints import file_hash, read_checkpoint
    from apexrl.rl.promotion import qualifies
    for name in ("all-circuits-v1", "rookie-reliable-v1"):
        path, manifest, _ = read_checkpoint(ROOT/f"models/{name}.zip", ROOT)
        report = json.loads(path.with_suffix(".evaluation.json").read_text(encoding="utf-8"))
        if not qualifies(report, .9) or report["model_sha256"] != manifest["model_sha256"]:
            raise SystemExit(f"Published driver {name} failed release validation")
    work = ROOT/"build/windows"
    work.mkdir(parents=True, exist_ok=True)
    source_data = work/"core"
    source_data.mkdir(exist_ok=True)
    for file in (ROOT/"apexrl/core").glob("*.py"):
        target = source_data/file.name
        if not target.exists() or target.read_bytes() != file.read_bytes():
            shutil.copy2(file, target)
    import pygame
    from PIL import Image
    from apexrl.game.identity import app_icon
    icon = work/"ApexRL.ico"
    surface = app_icon(256)
    Image.frombytes("RGBA", (256,256), pygame.image.tobytes(surface, "RGBA")).save(icon, sizes=[(32,32),(64,64),(128,128),(256,256)])
    arguments = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir", "--windowed",
                 "--name", "ApexRL", "--icon", str(icon), "--distpath", str(ROOT/"dist"),
                 "--workpath", str(work/"analysis"), "--specpath", str(work),
                 "--add-data", f"{ROOT/'config'};config", "--add-data", f"{ROOT/'tracks'};tracks",
                 "--add-data", f"{ROOT/'models'};models", "--add-data", f"{source_data};apexrl/core",
                 "--hidden-import", "stable_baselines3", "--collect-data", "stable_baselines3"]
    for package in ("torch", "gymnasium", "stable-baselines3", "numpy", "pygame"):
        arguments.extend(("--copy-metadata", package))
    for package in ("pytest", "IPython", "tkinter", "tensorboard", "matplotlib"):
        arguments.extend(("--exclude-module", package))
    arguments.append(str(ROOT/"main.py"))
    # PyInstaller replaces its output directory. Retain portable user saves.
    save_directory = ROOT/"dist/ApexRL/saves"
    preserved_saves = {file.relative_to(save_directory):file.read_bytes()
                       for file in save_directory.rglob("*") if file.is_file()} if save_directory.exists() else {}
    subprocess.run(arguments, cwd=ROOT, check=True)
    directory = ROOT/"dist/ApexRL"
    for relative, contents in preserved_saves.items():
        target = directory/"saves"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    shutil.copyfile(ROOT/"RELEASE.md", directory/"START-HERE.md")
    if (ROOT/"RELEASE_VALIDATION.md").exists():
        shutil.copyfile(ROOT/"RELEASE_VALIDATION.md", directory/"RELEASE_VALIDATION.md")
    collect_notices(directory/"licenses")
    report_path = work/"packaged-verification.json"
    subprocess.run([str(directory/"ApexRL.exe"), "--self-check", "--output", str(report_path)], cwd=ROOT, check=True)
    verified = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {("all-circuits-v1.zip",track) for track in ("rookie_ring", "apex_valley", "velocity_circuit")}
    expected.add(("rookie-reliable-v1.zip", "rookie_ring"))
    actual = {(row["model"],row["track"]) for row in verified["drivers"] if row["success"]}
    if not verified["success"] or not verified["rl_available"] or actual != expected:
        raise SystemExit("Packaged verification failed; release archives were not updated")
    manifest = {"version":__version__, "platform":"windows-x64", "models":{
        path.name:file_hash(path) for path in sorted((directory/"_internal/models").glob("*.zip"))}}
    (directory/"release.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_directory(directory, ROOT/f"dist/ApexRL-{__version__}-Windows-x64.zip")
    source_zip = ROOT/f"dist/ApexRL-{__version__}-Source.zip"
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        files = list(ROOT.glob("*.py"))+list(ROOT.glob("*.md"))+list(ROOT.glob("requirements*.txt"))+[ROOT/"pyproject.toml",ROOT/".gitignore"]
        for name in ("apexrl", "config", "tracks", "tests", "models"):
            files.extend(path for path in (ROOT/name).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
        for file in sorted(set(files)):
            archive.write(file, "ApexRL-Source/"+file.relative_to(ROOT).as_posix())
    checksums = {file.name:file_hash(file) for file in (ROOT/"dist").glob(f"ApexRL-{__version__}-*.zip")}
    (ROOT/"dist/SHA256SUMS.txt").write_text("".join(f"{value}  {name}\n" for name,value in sorted(checksums.items())), encoding="utf-8")
    print(f"Built Windows and source packages in {ROOT/'dist'}", flush=True)


if __name__ == "__main__":
    main()
