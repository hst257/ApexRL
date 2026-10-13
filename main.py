"""Run with `python main.py`; bounded runs are available for visual smoke checks."""
import argparse
import os
from pathlib import Path
import sys

from apexrl.utils.runtime import resource_root


def main():
    parser = argparse.ArgumentParser(description="APEX RL — original simcade time trials")
    parser.add_argument("--track", choices=("rookie_ring", "apex_valley", "velocity_circuit"))
    parser.add_argument("--frames", type=int, help="Close after this many rendered frames")
    parser.add_argument("--screenshot", help="Save the final logical viewport to a PNG")
    parser.add_argument("--ai-model", type=Path, help="Watch a trusted local PPO checkpoint in the game")
    parser.add_argument("--ai-garage", action="store_true", help="Open the AI policy selector")
    parser.add_argument("--self-check", action="store_true", help="Verify packaged gameplay and published drivers")
    parser.add_argument("--output", type=Path, help="Write the self-check JSON report here")
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("frames must be positive")
    root = resource_root()
    if args.self_check:
        if args.output is None:
            parser.error("--self-check requires --output")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        from apexrl.utils.release_check import verify_release
        report = verify_release(root, args.output)
        return 0 if report["success"] else 1
    from apexrl.game.application import Application
    app = Application(root)
    if args.track:
        app.start_race(next(i for i, track in enumerate(app.tracks) if track.id == args.track))
    if args.ai_model:
        from apexrl.game.policies import PolicyEntry
        app.activate("ai")
        model_path = args.ai_model if args.ai_model.is_absolute() else root/args.ai_model
        app.ai_models = [PolicyEntry(model_path, model_path.stem, "unknown", 0)]
        app.ai_model_index = 0
        if not app.start_ai():
            print(app.ai_error)
    elif args.ai_garage:
        app.activate("ai")
    app.run(args.frames, args.screenshot)
    return 0


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    if getattr(sys, "frozen", False):
        from apexrl.utils.runtime import default_save_directory
        logs = default_save_directory(resource_root())
        logs.mkdir(parents=True, exist_ok=True)
        log = (logs/"diagnostics.log").open("a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log
    raise SystemExit(main())
