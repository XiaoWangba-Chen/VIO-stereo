#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(path):
    rows = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if len(row) < 11:
                continue
            try:
                rows.append([float(value) for value in row[:11]])
            except ValueError:
                continue
    trajectory = np.asarray(rows, dtype=np.float64)
    if not len(trajectory) or not np.isfinite(trajectory).all():
        raise SystemExit("trajectory is empty or contains NaN/Inf")
    trajectory[:, 0] /= 1e9
    trajectory[:, 0] -= trajectory[0, 0]
    return trajectory


def percentile(values, quantile):
    return float(np.percentile(values, quantile))


def write_metrics(path, trajectory):
    position = trajectory[:, 1:4]
    velocity = trajectory[:, 8:11]
    steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
    speeds = np.linalg.norm(velocity, axis=1)
    norms = np.linalg.norm(position, axis=1)
    metrics = {
        "states": len(trajectory),
        "duration_s": trajectory[-1, 0],
        "final_x_m": position[-1, 0],
        "final_y_m": position[-1, 1],
        "final_z_m": position[-1, 2],
        "final_distance_from_origin_m": norms[-1],
        "maximum_distance_from_origin_m": norms.max(),
        "integrated_path_length_m": steps.sum(),
        "median_speed_mps": np.median(speeds),
        "p95_speed_mps": percentile(speeds, 95),
        "maximum_speed_mps": speeds.max(),
        "median_position_step_m": np.median(steps),
        "maximum_position_step_m": steps.max(),
        "timestamp_gaps_over_0_1s": int(np.count_nonzero(np.diff(trajectory[:, 0]) > 0.1)),
    }
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows(metrics.items())
    return metrics


def plot(path, trajectory, title, time_limit=None):
    if time_limit is not None:
        trajectory = trajectory[trajectory[:, 0] <= time_limit]
    time = trajectory[:, 0]
    position = trajectory[:, 1:4]
    speed = np.linalg.norm(trajectory[:, 8:11], axis=1)
    distance = np.linalg.norm(position, axis=1)

    figure = plt.figure(figsize=(13, 10))
    xy = figure.add_subplot(2, 2, 1)
    xz = figure.add_subplot(2, 2, 2)
    xyz = figure.add_subplot(2, 2, 3, projection="3d")
    trend = figure.add_subplot(2, 2, 4)

    xy.plot(position[:, 0], position[:, 1], linewidth=0.9)
    xz.plot(position[:, 0], position[:, 2], linewidth=0.9)
    xyz.plot(position[:, 0], position[:, 1], position[:, 2], linewidth=0.8)
    for axis, xlabel, ylabel, subtitle in (
            (xy, "x [m]", "y [m]", "XY trajectory"),
            (xz, "x [m]", "z [m]", "XZ trajectory")):
        axis.scatter(position[0, 0], position[0, 1 if axis is xy else 2], color="green", label="start")
        axis.scatter(position[-1, 0], position[-1, 1 if axis is xy else 2], color="red", label="end")
        axis.set(xlabel=xlabel, ylabel=ylabel, title=subtitle)
        axis.axis("equal")
        axis.grid(alpha=0.25)
        axis.legend()
    xyz.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="3D trajectory")

    trend.plot(time, distance, color="tab:blue", label="distance from origin")
    trend.set(xlabel="elapsed time [s]", ylabel="distance [m]", title="Position magnitude and speed")
    trend.grid(alpha=0.25)
    speed_axis = trend.twinx()
    speed_axis.plot(time, speed, color="tab:orange", alpha=0.7, label="speed")
    speed_axis.set_ylabel("speed [m/s]")
    lines = trend.lines + speed_axis.lines
    trend.legend(lines, [line.get_label() for line in lines], loc="upper left")

    suffix = f" (first {time_limit:g} s)" if time_limit is not None else ""
    figure.suptitle(title + suffix)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="VINS trajectory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = load_trajectory(args.trajectory)
    metrics = write_metrics(args.output_dir / "trajectory_metrics.csv", trajectory)
    plot(args.output_dir / "trajectory_overview.png", trajectory, args.title)
    plot(args.output_dir / "trajectory_first_60s.png", trajectory, args.title, 60.0)
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
