"""Measure complete headless simulation updates, including snapshots and contacts."""
import argparse
import json
import math
import platform
from pathlib import Path
import time

from apexrl.core import RacingSimulation, VehicleControl, load_tracks
from apexrl.core.simulation import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--track", choices=("all", "rookie_ring", "apex_valley", "velocity_circuit"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 100_000:
        parser.error("At least 100,000 measured steps are required")
    report = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(), "results": []}
    actions = tuple(VehicleControl(steering=.28*math.sin(i*.006), throttle=.75, brake=.1 if i%900 > 750 else 0) for i in range(3600))
    for track in load_tracks(Path(__file__).resolve().parent/"tracks"):
        if args.track != "all" and track.id != args.track:
            continue
        simulation = RacingSimulation(track, seed=42, simulation_config=SimulationConfig(countdown_seconds=0))
        for i in range(1000):
            simulation.step(actions[i%len(actions)])
        simulation.reset()
        start = time.perf_counter()
        for i in range(args.steps):
            simulation.step(actions[i%len(actions)])
        elapsed = time.perf_counter()-start
        result = {"track": track.id, "steps": args.steps, "seconds": round(elapsed, 3), "steps_per_second": round(args.steps/elapsed, 1)}
        report["results"].append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
