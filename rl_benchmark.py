"""Measure the Gymnasium adapter, including observations/rewards and episode resets."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10000, help="Measured agent steps per track")
    parser.add_argument("--output", type=Path, default=Path("artifacts/rl-benchmark.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    from apexrl.core import load_tracks
    from apexrl.rl import ApexRacingEnv
    from apexrl.rl.environment import ROOT
    tracks = load_tracks(ROOT/"tracks")
    actions = np.random.default_rng(42).uniform(-1,1,(args.steps,2)).astype(np.float32)
    report = {"python":platform.python_version(), "platform":platform.platform(), "includes_neural_inference":False,"results":[]}
    for track in tracks:
        env = ApexRacingEnv(track.id, tracks=tracks)
        try:
            env.reset(seed=42)
            for action in actions[:min(100,args.steps)]:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    env.reset()
            env.reset(seed=42)
            ticks = episodes = 0
            start = time.perf_counter()
            for action in actions:
                _, _, terminated, truncated, info = env.step(action)
                ticks += info["physics_ticks"]
                if terminated or truncated:
                    episodes += 1
                    env.reset()
            elapsed = time.perf_counter()-start
            row = {"track":track.id,"agent_steps":args.steps,"physics_ticks":ticks,"episode_resets":episodes,
                   "seconds":elapsed,"agent_steps_per_second":args.steps/elapsed,"physics_ticks_per_second":ticks/elapsed}
            report["results"].append(row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
