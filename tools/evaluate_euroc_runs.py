#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("MH_01_easy", "MH_02_easy", "MH_03_medium")
MODES = ("mono", "stereo")


def read_csv(path, columns):
    rows = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append([float(row[index]) for index in columns])
    return np.asarray(rows, dtype=np.float64)


def interpolate_ground_truth(ground_truth, timestamps):
    valid = (timestamps >= ground_truth[0, 0]) & (timestamps <= ground_truth[-1, 0])
    timestamps = timestamps[valid]
    positions = np.column_stack([
        np.interp(timestamps, ground_truth[:, 0], ground_truth[:, axis])
        for axis in range(1, 4)
    ])
    return valid, timestamps, positions


def umeyama(source, target, estimate_scale):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1] = -1.0
    rotation = vt.T @ np.diag(correction) @ u.T
    scale = 1.0
    if estimate_scale:
        scale = float(np.sum(singular_values * correction) /
                      np.mean(np.sum(source_centered * source_centered, axis=1)))
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = scale * (rotation @ source.T).T + translation
    return aligned, scale


def rmse(vectors):
    return float(math.sqrt(np.mean(np.sum(vectors * vectors, axis=1))))


def rpe_translation(timestamps, estimate, truth, delta=1.0):
    errors = []
    for index, timestamp in enumerate(timestamps):
        other = int(np.searchsorted(timestamps, timestamp + delta))
        if other >= len(timestamps) or abs(timestamps[other] - timestamp - delta) > 0.06:
            continue
        errors.append((estimate[other] - estimate[index]) - (truth[other] - truth[index]))
    return rmse(np.asarray(errors)) if errors else float("nan")


def count_log_events(path):
    text = path.read_text(errors="ignore").replace("\r", "\n")
    backend = re.findall(
        r"stereo backend: left_obs=(\d+) right_obs=(\d+) stereo_landmarks=(\d+) "
        r"valid_inverse_depth=(\d+) left_median=([0-9.]+)px right_median=([0-9.]+)px "
        r"rejected_right=(\d+)", text)
    marginalization_counts = [
        int(value) for value in re.findall(
            r"right factors added to marginalization: (\d+)", text)
    ]
    return {
        "initializations": text.count("Initialization finish!"),
        "failures": text.count("failure detection!"),
        "reboots": text.count("system reboot!"),
        "metric_init_failed_attempts": text.count("stereo metric init failed"),
        "right_backend_samples": len(backend),
        "left_reprojection_median_px": (float(np.median([float(row[4]) for row in backend]))
                                         if backend else float("nan")),
        "right_reprojection_median_px": (float(np.median([float(row[5]) for row in backend]))
                                          if backend else float("nan")),
        "right_observations_median": (float(np.median([int(row[1]) for row in backend]))
                                      if backend else float("nan")),
        "logged_marginalized_right_factors_sum": sum(marginalization_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    output_root = args.repo / "outputs/euroc_eval"
    metrics = []
    trajectories = {}

    for dataset in DATASETS:
        ground_truth = read_csv(
            args.repo / "datasets" / dataset / "mav0/state_groundtruth_estimate0/data.csv",
            (0, 1, 2, 3))
        ground_truth[:, 0] /= 1e9
        camera = read_csv(args.repo / "datasets" / dataset / "mav0/cam0/data.csv", (0,))
        camera[:, 0] /= 1e9
        trajectories[dataset] = {}
        aligned_rows = []

        for mode in MODES:
            estimate = read_csv(output_root / dataset / mode / "vins_result_no_loop.csv", (0, 1, 2, 3))
            estimate[:, 0] /= 1e9
            valid, timestamps, truth = interpolate_ground_truth(ground_truth, estimate[:, 0])
            source = estimate[valid, 1:4]
            aligned_se3, _ = umeyama(source, truth, False)
            aligned_sim3, sim3_scale = umeyama(source, truth, True)
            events = count_log_events(output_root / dataset / mode / "run.log")
            duration = timestamps[-1] - timestamps[0]
            dataset_duration = camera[-1, 0] - camera[0, 0]
            metrics.append({
                "dataset": dataset,
                "mode": mode,
                "states": len(source),
                "coverage_percent": 100.0 * duration / dataset_duration,
                "init_latency_s": timestamps[0] - camera[0, 0],
                "ate_se3_rmse_m": rmse(aligned_se3 - truth),
                "ate_sim3_rmse_m": rmse(aligned_sim3 - truth),
                "sim3_scale": sim3_scale,
                "rpe_translation_1s_rmse_m": rpe_translation(timestamps, aligned_se3, truth),
                **events,
            })
            trajectories[dataset][mode] = (aligned_se3, truth)
            for stamp, gt, aligned in zip(timestamps, truth, aligned_se3):
                aligned_rows.append([mode, f"{stamp:.9f}", *gt, *aligned])

        aligned_path = output_root / dataset / "aligned_trajectories.csv"
        with aligned_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["mode", "timestamp_s", "gt_x", "gt_y", "gt_z",
                             "aligned_x", "aligned_y", "aligned_z"])
            writer.writerows(aligned_rows)

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        colors = {"mono": "tab:orange", "stereo": "tab:blue"}
        for mode in MODES:
            aligned, truth = trajectories[dataset][mode]
            axes[0].plot(aligned[:, 0], aligned[:, 1], color=colors[mode], label=mode, linewidth=1.0)
            axes[1].plot(aligned[:, 0], aligned[:, 2], color=colors[mode], label=mode, linewidth=1.0)
        reference = trajectories[dataset]["mono"][1]
        axes[0].plot(reference[:, 0], reference[:, 1], "k--", label="ground truth", linewidth=1.0)
        axes[1].plot(reference[:, 0], reference[:, 2], "k--", label="ground truth", linewidth=1.0)
        axes[0].set(xlabel="x [m]", ylabel="y [m]", title="XY")
        axes[1].set(xlabel="x [m]", ylabel="z [m]", title="XZ")
        for axis in axes:
            axis.axis("equal")
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle(dataset + " SE(3)-aligned trajectories")
        figure.tight_layout()
        figure.savefig(output_root / dataset / "trajectory_comparison.png", dpi=160)
        plt.close(figure)

    metric_path = output_root / "metrics.csv"
    with metric_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(DATASETS))
    width = 0.36
    for offset, mode, color in ((-width / 2, "mono", "tab:orange"),
                                (width / 2, "stereo", "tab:blue")):
        values = [next(row["ate_se3_rmse_m"] for row in metrics
                       if row["dataset"] == dataset and row["mode"] == mode)
                  for dataset in DATASETS]
        axis.bar(x + offset, values, width, label=mode, color=color)
    axis.set_xticks(x, DATASETS)
    axis.set_ylabel("ATE RMSE [m], SE(3) aligned")
    axis.set_title("EuRoC trajectory accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "ate_comparison.png", dpi=160)
    plt.close(figure)

    for row in metrics:
        print(row)


if __name__ == "__main__":
    main()
