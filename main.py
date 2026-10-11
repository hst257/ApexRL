"""Run with `python main.py`; bounded runs are available for visual smoke checks."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="APEX RL — original simcade time trials")
    parser.add_argument("--track", choices=("rookie_ring", "apex_valley", "velocity_circuit"))
    parser.add_argument("--frames", type=int, help="Close after this many rendered frames")
    parser.add_argument("--screenshot", help="Save the final logical viewport to a PNG")
    args = parser.parse_args()
    from apexrl.game.application import Application
    app = Application(Path(__file__).resolve().parent)
    if args.track:
        app.start_race(next(i for i, track in enumerate(app.tracks) if track.id == args.track))
    app.run(args.frames, args.screenshot)


if __name__ == "__main__":
    main()
