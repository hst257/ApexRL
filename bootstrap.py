"""Teach a local PPO actor from conservative demonstrations on all circuits."""
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root/"config/training-all.json")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--episodes-per-track", type=int, default=3)
    parser.add_argument("--updates", type=int, default=2500)
    args = parser.parse_args()
    if min(args.rounds, args.episodes_per_track, args.updates) < 1:
        parser.error("Require positive rounds, episodes, and updates")
    from apexrl.rl.bootstrap import bootstrap
    from apexrl.rl.config import load_training_config
    bootstrap(root=root, config=load_training_config(args.config), source=args.initialize_from,
              run_dir=args.run_dir, rounds=args.rounds, episodes_per_track=args.episodes_per_track, updates=args.updates)


if __name__ == "__main__":
    main()
