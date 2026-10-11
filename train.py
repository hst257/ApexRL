"""Train the optional Phase 2 PPO policy; run with the project virtual environment."""
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root/"config/training.json")
    parser.add_argument("--track", choices=("rookie_ring", "apex_valley", "velocity_circuit", "all"), default="rookie_ring")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--vector", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--smoke", action="store_true", help="16,384 steps; two envs and two evaluation episodes")
    args = parser.parse_args()
    from apexrl.rl.config import load_training_config
    from apexrl.rl.training import train
    config = load_training_config(args.config)
    if args.smoke:
        config.update(total_timesteps=16384, n_envs=2, evaluation_episodes=2, checkpoint_frequency=8192, evaluation_frequency=8192)
    for key, value in (("total_timesteps", args.timesteps), ("n_envs", args.n_envs), ("seed", args.seed)):
        if value is not None:
            if value < (0 if key == "seed" else 1):
                parser.error(f"Invalid {key}")
            config[key] = value
    train(root=root, config=config, track_id=args.track, run_dir=args.run_dir,
          resume=args.resume, vector=args.vector, device=args.device)


if __name__ == "__main__":
    main()
