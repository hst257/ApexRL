"""Evaluate a trusted checkpoint on held-out seeds and publish a local AI Garage model."""
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--track", choices=("rookie_ring", "apex_valley", "velocity_circuit", "all"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=200042)
    parser.add_argument("--minimum-success", type=float, default=.9)
    args = parser.parse_args()
    if args.episodes < 5 or args.seed < 0 or not 0 < args.minimum_success <= 1:
        parser.error("Require at least five episodes, a nonnegative seed, and success threshold in (0, 1]")
    import torch
    from stable_baselines3 import PPO
    from apexrl.rl.checkpoints import read_checkpoint, write_json
    from apexrl.rl.evaluation import evaluate
    from apexrl.rl.promotion import publish_policy, qualifies, validate_name
    try:
        validate_name(args.name)
    except ValueError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parent
    torch.set_num_threads(1)
    checked, metadata, config = read_checkpoint(args.model, root)
    training_seed = metadata["training_config"]["seed"]
    intervals = [(training_seed, metadata["training_config"]["n_envs"]),
                 (training_seed+100000, metadata["training_config"]["evaluation_episodes"])]
    if any(args.seed < start+count and args.seed+args.episodes > start for start, count in intervals):
        parser.error("Choose held-out seeds distinct from training and scheduled evaluation seeds")
    model = PPO.load(checked, device="cpu")
    report = evaluate(model, root=root, config=config, track_id=args.track or metadata["training_track"],
                      episodes=args.episodes, seed=args.seed, collect_trajectories=True)
    report.update(model=str(checked.resolve()), model_sha256=metadata["model_sha256"], num_timesteps=metadata["num_timesteps"])
    report_path = checked.parent/f"promotion-{args.name}.json"
    write_json(report_path, report)
    for track, result in report["summary"].items():
        print(f"{track}: held-out valid laps {result['success_rate']:.0%}, best lap {result['best_lap_seconds']}")
    if not qualifies(report, args.minimum_success):
        print(f"Policy not published; held-out evaluation saved to {report_path.resolve()}")
        return 2
    published = publish_policy(checked, root, args.name, report, args.minimum_success)
    print(f"Published local AI Garage model: {published.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
