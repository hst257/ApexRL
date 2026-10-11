from collections import Counter
import numpy as np
from .environment import ApexRacingEnv, TRACK_IDS


def evaluate(model, *, root, config, track_id="all", episodes=3, seed=42, tracks=None,
             canonical_start=False, collect_trajectories=False):
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative")
    track_ids = TRACK_IDS if track_id == "all" else (track_id,)
    rows, summaries, trajectories = [], {}, []
    for identifier in track_ids:
        env = ApexRacingEnv(identifier, root=root, config=config, tracks=tracks)
        local_rows = []
        try:
            for episode in range(episodes):
                episode_seed = seed+TRACK_IDS.index(identifier)*10000+episode
                observation, info = env.reset(seed=episode_seed, options={"evaluation": canonical_start})
                total_reward = max_progress = steps = 0
                speeds = []
                trace = []
                terminated = truncated = False
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    max_progress = max(max_progress, info["progress_fraction"])
                    steps += 1
                    speeds.append(info["speed_kmh"])
                    if collect_trajectories and episode == 0:
                        vehicle = env.simulation.snapshot().vehicle
                        trace.append([vehicle.x, vehicle.y, vehicle.heading, info["speed_kmh"]])
                row = {"track": identifier, "episode": episode+1, "seed": episode_seed, "return": total_reward,
                       "steps": steps, "seconds": info["episode_seconds"], "max_progress_fraction": max_progress,
                       "mean_speed_kmh": float(np.mean(speeds)), "is_success": info["is_success"],
                       "lap_time": info["lap_time"], "termination_reason": info["termination_reason"],
                       "terminated": terminated, "truncated": truncated}
                rows.append(row)
                local_rows.append(row)
                if trace:
                    trajectories.append({"track": identifier, "episode_seed": episode_seed, "points": trace})
        finally:
            env.close()
        lap_times = [row["lap_time"] for row in local_rows if row["is_success"]]
        summaries[identifier] = {"episodes": episodes, "success_rate": sum(row["is_success"] for row in local_rows)/episodes,
                                 "mean_return": float(np.mean([row["return"] for row in local_rows])),
                                 "mean_max_progress_fraction": float(np.mean([row["max_progress_fraction"] for row in local_rows])),
                                 "best_lap_seconds": min(lap_times) if lap_times else None,
                                 "termination_counts": dict(Counter(row["termination_reason"] for row in local_rows))}
    return {"seed": seed, "canonical_start": canonical_start, "deterministic_policy": True,
            "summary": summaries, "episodes": rows, "trajectories": trajectories}
