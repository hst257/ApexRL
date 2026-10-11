"""Export learning curves and evaluated driving trajectories from a recorded run."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np


def rolling(values, window=20):
    return np.array([np.mean(values[max(0, index-window+1):index+1]) for index in range(len(values))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str((args.run_dir/".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"#f2f4ed", "axes.facecolor":"#f2f4ed", "font.family":"DejaVu Sans",
                         "axes.spines.top":False, "axes.spines.right":False, "axes.labelcolor":"#1c3335"})
    with (args.run_dir/"episodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("Run contains no completed episodes to plot")
    timesteps = [int(row["timesteps"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15,4.3), constrained_layout=True)
    for ax, key, title, scale in zip(axes, ("return", "max_progress", "success"),
                                     ("Episode return", "Maximum lap progress (%)", "Valid lap success (%)"), (1,100,100)):
        values = np.array([float(row[key])*scale for row in rows])
        ax.scatter(timesteps, values, color="#589386", s=8, alpha=.22)
        ax.plot(timesteps, rolling(values), color="#163f45", linewidth=2, label="Rolling mean / 20 episodes")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Environment timesteps")
        ax.grid(axis="y", alpha=.15)
        if key in ("max_progress", "success"):
            ax.set_ylim(-2,102)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("APEX RL / PPO learning history", x=.025, ha="left", fontsize=18, fontweight="bold")
    figure.savefig(args.run_dir/"learning_curves.png", dpi=160)
    plt.close(figure)
    report_path = args.run_dir/"final_evaluation.json"
    if report_path.exists():
        from apexrl.core import load_tracks
        tracks = load_tracks(Path(__file__).resolve().parent/"tracks")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        traces = {trace["track"]: trace["points"] for trace in report.get("trajectories", [])}
        figure, axes = plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
        for ax, track in zip(axes, tracks):
            points = np.vstack((track.points, track.points[0]))
            ax.plot(points[:,0], points[:,1], color="#b7c4bd", linewidth=5)
            ax.plot(points[:,0], points[:,1], color="#455f5b", linewidth=1)
            if track.id in traces:
                trace = np.array(traces[track.id])
                ax.plot(trace[:,0], trace[:,1], color="#cf694d", linewidth=2, label="First evaluation episode")
                ax.scatter(trace[-1,0],trace[-1,1],color="#cf694d",s=25)
            ax.scatter(*track.finish.center, color="#296051", marker="s",s=30)
            summary = report["summary"][track.id]
            ax.set_title(f"{track.name}\nMean progress {summary['mean_max_progress_fraction']:.1%} / valid laps {summary['success_rate']:.0%}",loc="left",fontweight="bold")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xlabel("World X (meters)")
            ax.set_ylabel("World Y (meters)")
        figure.suptitle("Measured policy trajectories / no inferred lap times", x=.025,ha="left",fontsize=17,fontweight="bold")
        figure.savefig(args.run_dir/"evaluation_trajectories.png",dpi=160)
        plt.close(figure)
    print(f"Plots written to {args.run_dir.resolve()}")


if __name__ == "__main__":
    main()
