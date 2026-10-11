"""Scripted presentation checks. This is not a hands-on driving verification."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/visual"))
    parser.add_argument("--native", action="store_true", help="Use the real desktop display instead of SDL's test driver")
    args = parser.parse_args()
    if not args.native:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    import pygame
    from apexrl.core import VehicleControl
    from apexrl.game.application import Application
    root = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    app = Application(root, args.output/"saves")
    summary = {"display_driver": pygame.display.get_driver(), "screenshots": [], "render_fps": {}}
    def capture(name):
        app.draw()
        path = args.output/f"{name}.png"
        pygame.image.save(app.canvas, str(path))
        summary["screenshots"].append(str(path.resolve()))
    try:
        capture("main")
        for screen in ("tracks", "controls", "settings"):
            app.transition(screen)
            capture(screen)
        for index, track in enumerate(app.tracks):
            app.debug = app.expanded = False
            app.start_race(index)
            app.advance(1/60, VehicleControl())
            capture(track.id+"_countdown")
            for _ in range(280):
                app.advance(1/60, VehicleControl(throttle=.8, steering=.03))
            capture(track.id+"_race")
            app.expanded = True
            capture(track.id+"_overview")
            app.expanded = False
            app.debug = True
            capture(track.id+"_debug")
            app.debug = False
            start = time.perf_counter()
            for _ in range(120):
                app.draw()
            summary["render_fps"][track.id] = round(120/(time.perf_counter()-start), 1)
        app.transition("pause")
        capture("pause")
        app.event(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=640))
        app.draw()
        pygame.image.save(app.display, str(args.output/"resized.png"))
        summary["screenshots"].append(str((args.output/"resized.png").resolve()))
        (args.output/"report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
