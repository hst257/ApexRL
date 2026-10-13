"""Exercise the packaged resources, game loop, and every published driver's laps."""
import json
from pathlib import Path


def verify_release(root: Path, output: Path) -> dict:
    import pygame
    from apexrl.core import VehicleControl
    from apexrl.game.application import Application
    from apexrl.game.policies import discover_policies, rl_available

    output.parent.mkdir(parents=True, exist_ok=True)
    app = Application(root, output.parent/"verification-saves")
    report = {"manual": [], "drivers": [], "rl_available": rl_available(), "success": False}
    records = dict(app.records.records)
    try:
        for index, track in enumerate(app.tracks):
            app.start_race(index)
            for _ in range(240):
                app.advance(1/60, VehicleControl(throttle=.8))
            assert app.state.vehicle.speed > 0
            before = app.state
            app.event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
            app.advance(.5)
            assert app.state == before
            app.draw()
            report["manual"].append({"track": track.id, "acceleration": True, "pause": True})
        if report["rl_available"]:
            app.event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
            for entry in discover_policies(root):
                if not entry.path.is_relative_to(root/"models"):
                    continue
                for track_id, _, success_rate in entry.validation:
                    if success_rate < .9:
                        continue
                    app.current_track = next(i for i, track in enumerate(app.tracks) if track.id == track_id)
                    app.ai_models = [entry]
                    app.ai_model_index = 0
                    assert app.start_ai(), app.ai_error
                    for frame in range(6000):
                        app.advance(1/30)
                        if frame % 300 == 0:
                            pygame.event.pump()
                            app.draw()
                        if app.ai_session.finished:
                            break
                    info = app.ai_session.info
                    row = {"model":entry.path.name, "track":track_id, "success":bool(info["is_success"]),
                           "lap_time":info["lap_time"], "reason":info["termination_reason"]}
                    report["drivers"].append(row)
                    assert row["success"], row
                    app.draw()
                    pygame.image.save(app.canvas, str(output.parent/f"{entry.path.stem}-{track_id}.png"))
        assert app.records.records == records
        report["human_records_unchanged"] = True
        report["success"] = True
    except Exception as error:
        report["error"] = str(error)
    finally:
        app.close()
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
