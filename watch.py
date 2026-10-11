"""Watch a saved AI policy without changing human lap records."""
import argparse
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--track", choices=("rookie_ring", "apex_valley", "velocity_circuit"), default="rookie_ring")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, help="Limit rendered agent steps for a smoke launch")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or (args.frames is not None and args.frames < 1):
        parser.error("episodes/frames must be positive; seed must be nonnegative")
    import torch
    import pygame
    from stable_baselines3 import PPO
    from apexrl.rl.checkpoints import read_checkpoint
    from apexrl.rl.environment import ApexRacingEnv
    torch.set_num_threads(1)
    path, _, config = read_checkpoint(args.model, root)
    model = PPO.load(path, device="cpu")
    env = ApexRacingEnv(args.track, root=root, config=config, render_mode="human")
    frames = 0
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed+episode, options={"evaluation": True})
            terminated = truncated = False
            while not (terminated or truncated) and not env._viewer.close_requested:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                frames += 1
                if args.frames and frames >= args.frames:
                    break
            print(f"Episode {episode+1}: {info['termination_reason'] or 'viewer stopped'}, progress {info['progress_fraction']:.1%}")
            if env._viewer.close_requested or (args.frames and frames >= args.frames):
                break
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(env._viewer.canvas, str(args.screenshot))
    finally:
        env.close()


if __name__ == "__main__":
    main()
