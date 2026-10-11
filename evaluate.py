"""Evaluate a saved deterministic policy on seeded episodes and write measured metrics."""
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--track", choices=("rookie_ring", "apex_valley", "velocity_circuit", "all"), default="all")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--canonical-start", action="store_true", help="Disable small seeded grid-pose perturbations")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive; seed must be nonnegative")
    import torch
    from stable_baselines3 import PPO
    from apexrl.rl.checkpoints import read_checkpoint, write_json
    from apexrl.rl.evaluation import evaluate
    torch.set_num_threads(1)
    path, metadata, config = read_checkpoint(args.model, root)
    model = PPO.load(path, device="cpu")
    result = evaluate(model, root=root, config=config, track_id=args.track, episodes=args.episodes, seed=args.seed,
                      canonical_start=args.canonical_start, collect_trajectories=True)
    result["model"] = str(path.resolve())
    result["model_sha256"] = metadata["model_sha256"]
    result["num_timesteps"] = metadata["num_timesteps"]
    output = args.output or path.parent/"evaluation.json"
    write_json(output, result)
    for name, summary in result["summary"].items():
        print(f"{name}: success {summary['success_rate']:.0%}, mean progress {summary['mean_max_progress_fraction']:.1%}, best valid lap {summary['best_lap_seconds']}")
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    main()
