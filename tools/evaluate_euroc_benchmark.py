#!/usr/bin/env python3
"""Evaluate all successful runs produced by run_euroc_benchmark.py."""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from euroc_benchmark_metrics import (interpolate_quaternions, percentile,
                                     quat_multiply, read_numeric_csv,
                                     relative_pose_errors, rmse,
                                     rotation_matrix_to_quat, umeyama,
                                     vector_rmse)


def dict_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite(values):
    return [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]


def describe(prefix, values):
    values = finite(values)
    if not values:
        return {prefix + suffix: float("nan") for suffix in ("_mean", "_p50", "_p90", "_p95", "_max")}
    return {prefix + "_mean": float(np.mean(values)),
            prefix + "_p50": percentile(values, 50),
            prefix + "_p90": percentile(values, 90),
            prefix + "_p95": percentile(values, 95),
            prefix + "_max": max(values)}


def read_trajectory(path):
    data = read_numeric_csv(path, tuple(range(8)))
    if data.ndim != 2 or data.shape[1] != 8 or len(data) < 2:
        raise ValueError("trajectory has fewer than two valid poses")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("trajectory timestamps are not strictly increasing")
    data[:, 0] /= 1e9
    data[:, 4:8] = data[:, [4, 5, 6, 7]]  # already qw,qx,qy,qz
    return data


def ground_truth_at(dataset, timestamps):
    ground = read_numeric_csv(dataset / "mav0/state_groundtruth_estimate0/data.csv", tuple(range(8)))
    ground[:, 0] /= 1e9
    valid = (timestamps >= ground[0, 0]) & (timestamps <= ground[-1, 0])
    selected = timestamps[valid]
    positions = np.column_stack([np.interp(selected, ground[:, 0], ground[:, axis]) for axis in range(1, 4)])
    quaternions = interpolate_quaternions(ground[:, 0], ground[:, 4:8], selected)
    return valid, selected, positions, quaternions


def resource_metrics(path):
    rows = dict_rows(path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    metrics = {}
    for name in ("feature_tracker", "vins_estimator", "roscore", "player"):
        entries = grouped.get(name, [])
        metrics.update(describe(name + "_cpu_percent", [row["cpu_percent"] for row in entries]))
        rss_mib = [float(row["rss_bytes"]) / 1048576.0 for row in entries]
        metrics.update(describe(name + "_rss_mib", rss_mib))
        if len(entries) > 1:
            elapsed = np.asarray([float(row["elapsed_s"]) for row in entries])
            cpu = np.asarray([float(row["cpu_percent"]) for row in entries])
            metrics[name + "_cpu_core_seconds"] = float(np.trapz(cpu / 100.0, elapsed))
        else:
            metrics[name + "_cpu_core_seconds"] = float("nan")
    core_names = ("feature_tracker", "vins_estimator", "roscore")
    metrics["vins_cpu_core_seconds"] = sum(metrics[name + "_cpu_core_seconds"] for name in core_names)
    # Samples are emitted consecutively for each group, so index alignment is deterministic.
    count = min((len(grouped.get(name, [])) for name in core_names), default=0)
    total_cpu, total_rss = [], []
    for index in range(count):
        total_cpu.append(sum(float(grouped[name][index]["cpu_percent"]) for name in core_names))
        total_rss.append(sum(float(grouped[name][index]["rss_bytes"]) for name in core_names) / 1048576.0)
    metrics.update(describe("vins_cpu_percent", total_cpu))
    metrics.update(describe("vins_rss_mib", total_rss))
    return metrics


def parse_final_sync(text):
    match = re.search(r"FINAL stereo sync: left=(\d+) right=(\d+) paired=(\d+) "
                      r"left_unpaired=(\d+) right_unpaired=(\d+) out_of_order=(\d+) "
                      r"overflow=(\d+) restarts=(\d+) max_queue=(\d+)", text)
    names = ("left_input", "right_input", "stereo_pairs", "left_unpaired", "right_unpaired",
             "out_of_order", "sync_overflow", "tracker_restarts", "stereo_sync_queue_max")
    return dict(zip(names, map(int, match.groups()))) if match else {}


def evaluate_run(dataset, mode, run_dir):
    trajectory = read_trajectory(run_dir / "vins_result_no_loop.csv")
    valid, timestamps, truth_p, truth_q = ground_truth_at(dataset, trajectory[:, 0])
    estimate_p, estimate_q = trajectory[valid, 1:4], trajectory[valid, 4:8]
    aligned_p, align_rotation, _, _ = umeyama(estimate_p, truth_p, False)
    sim3_p, _, _, sim3_scale = umeyama(estimate_p, truth_p, True)
    align_q = rotation_matrix_to_quat(align_rotation)
    aligned_q = quat_multiply(np.broadcast_to(align_q, estimate_q.shape), estimate_q)
    translation_rpe, rotation_rpe = relative_pose_errors(
        timestamps, aligned_p, aligned_q, truth_p, truth_q)

    camera = read_numeric_csv(dataset / "mav0/cam0/data.csv", (0,))[:, 0] / 1e9
    right_camera = read_numeric_csv(dataset / "mav0/cam1/data.csv", (0,))[:, 0]
    imu_input = read_numeric_csv(dataset / "mav0/imu0/data.csv", (0,))[:, 0]
    feature_rows = dict_rows(run_dir / "feature_timing.csv")
    backend_rows = dict_rows(run_dir / "backend_timing.csv")
    published = {row["timestamp_ns"]: row for row in feature_rows if row["estimator_published"] == "1"}
    backend_by_stamp = {row["timestamp_ns"]: row for row in backend_rows}
    joined = [(published[stamp], backend_by_stamp[stamp]) for stamp in published.keys() & backend_by_stamp.keys()]
    end_to_end = [(float(back["backend_finish_monotonic_ns"]) - float(front["frame_start_monotonic_ns"])) / 1e6
                  for front, back in joined]
    keyframe_lag = [(float(back["backend_finish_monotonic_ns"]) - float(front["frame_start_monotonic_ns"])) / 1e6
                    for front, back in joined if back["keyframe"] == "1"]
    estimator_log = (run_dir / "vins_estimator.log").read_text(errors="ignore")
    tracker_log = (run_dir / "feature_tracker.log").read_text(errors="ignore")
    missing_backend = max(0, len(published) - len(backend_rows))
    startup_skipped = min(1, missing_backend)  # estimator intentionally discards its first feature message
    metrics = {
        "mode": mode,
        "run": run_dir.name,
        "success": 1,
        "states": len(estimate_p),
        "coverage_percent": 100.0 * (timestamps[-1] - timestamps[0]) / (camera[-1] - camera[0]),
        "init_latency_s": timestamps[0] - camera[0],
        "ate_se3_rmse_m": vector_rmse(aligned_p - truth_p),
        "ate_sim3_rmse_m": vector_rmse(sim3_p - truth_p),
        "sim3_scale": sim3_scale,
        "rpe_translation_1s_rmse_m": rmse(translation_rpe),
        "rpe_translation_1s_median_m": percentile(translation_rpe, 50),
        "rpe_translation_1s_p95_m": percentile(translation_rpe, 95),
        "rpe_rotation_1s_rmse_rad": rmse(rotation_rpe),
        "rpe_rotation_1s_median_rad": percentile(rotation_rpe, 50),
        "rpe_rotation_1s_p95_rad": percentile(rotation_rpe, 95),
        "initializations": estimator_log.count("Initialization finish!"),
        "failures": estimator_log.count("failure detection!"),
        "reboots": estimator_log.count("system reboot!"),
        "feature_frames": len(feature_rows),
        "cam0_input": len(camera),
        "cam1_input": len(right_camera) if mode == "stereo" else 0,
        "imu_input": len(imu_input),
        "feature_published": len(published),
        "backend_processed": len(backend_rows),
        "startup_skipped_frames": startup_skipped,
        "dropped_frames": missing_backend - startup_skipped,
        "keyframes": sum(row["keyframe"] == "1" for row in backend_rows),
        "keyframe_rate_percent": 100.0 * sum(row["keyframe"] == "1" for row in backend_rows) / len(backend_rows),
        "window_size_max": max(int(row["window_states"]) for row in backend_rows),
        "stereo_queue_depth_max": max(int(row["stereo_queue_depth"]) for row in feature_rows),
    }
    metrics.update(parse_final_sync(tracker_log))
    metrics.update(describe("feature_tracking_ms", [row["feature_tracking_ms"] for row in feature_rows]))
    metrics.update(describe("stereo_matching_ms", [row["stereo_matching_ms"] for row in feature_rows]))
    metrics.update(describe("optimization_ms", [row["optimization_ms"] for row in backend_rows]))
    metrics.update(describe("marginalization_ms", [row["marginalization_ms"] for row in backend_rows]))
    metrics.update(describe("backend_total_ms", [row["backend_total_ms"] for row in backend_rows]))
    metrics.update(describe("end_to_end_latency_ms", end_to_end))
    metrics.update(describe("keyframe_backend_lag_ms", keyframe_lag))
    metrics.update(describe("imu_span_s", [row["imu_span_s"] for row in backend_rows]))
    metrics.update(resource_metrics(run_dir / "resources.csv"))
    integrity = (metrics["coverage_percent"] >= 98.0 and metrics["reboots"] == 0 and
                 metrics["failures"] == 0 and metrics["dropped_frames"] == 0)
    if mode == "stereo":
        expected_pairs = len(read_numeric_csv(dataset / "mav0/cam0/data.csv", (0,)))
        integrity = integrity and metrics.get("stereo_pairs", 0) == expected_pairs
    metrics["integrity_pass"] = int(integrity)
    metrics["realtime_pass"] = int(metrics["end_to_end_latency_ms_p95"] <= 100.0)
    aligned_csv = run_dir / "aligned_trajectory.csv"
    with aligned_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("timestamp_s", "gt_x", "gt_y", "gt_z", "aligned_x", "aligned_y", "aligned_z"))
        writer.writerows(np.column_stack((timestamps, truth_p, aligned_p)))
    return metrics, (truth_p, aligned_p)


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = []
    identity = {"mode", "run"}
    for mode in ("mono", "stereo"):
        selected = [row for row in rows if row["mode"] == mode]
        for metric in rows[0]:
            if metric in identity:
                continue
            values = finite(row.get(metric) for row in selected)
            if not values:
                continue
            result.append({"mode": mode, "metric": metric, "median": percentile(values, 50),
                           "min": min(values), "max": max(values),
                           "p90": percentile(values, 90), "p95": percentile(values, 95)})
    return result


def write_comparison(path, rows):
    important = ("ate_se3_rmse_m", "rpe_translation_1s_rmse_m", "rpe_rotation_1s_rmse_rad",
                 "feature_tracking_ms_p50", "stereo_matching_ms_p50", "optimization_ms_p50",
                 "end_to_end_latency_ms_p95", "vins_cpu_percent_mean", "vins_rss_mib_max",
                 "keyframe_rate_percent", "imu_span_s_p95", "dropped_frames")
    medians = {mode: {key: percentile([row[key] for row in rows if row["mode"] == mode], 50)
                      for key in important} for mode in ("mono", "stereo")}
    lines = ["# MH_03_medium mono/stereo comparison", "", "| Metric | Mono median | Stereo median | Difference | Change |",
             "|---|---:|---:|---:|---:|"]
    for key in important:
        mono, stereo = medians["mono"][key], medians["stereo"][key]
        change = 100.0 * (stereo - mono) / abs(mono) if mono else float("nan")
        lines.append(f"| {key} | {mono:.6g} | {stereo:.6g} | {stereo-mono:+.6g} | {change:+.2f}% |")
    lines += ["", "Integrity: %d/%d runs passed; real-time reference (P95 <= 100 ms): %d/%d runs passed." %
              (sum(row["integrity_pass"] for row in rows), len(rows),
               sum(row["realtime_pass"] for row in rows), len(rows)), ""]
    path.write_text("\n".join(lines))


def plot_results(output, rows, trajectories):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"mono": "tab:orange", "stereo": "tab:blue"}
    for mode, run, truth, aligned in trajectories:
        axes[0].plot(aligned[:, 0], aligned[:, 1], color=colors[mode], alpha=0.55, label=f"{mode}/{run}")
        axes[1].plot(aligned[:, 0], aligned[:, 2], color=colors[mode], alpha=0.55, label=f"{mode}/{run}")
    truth = trajectories[0][2]
    axes[0].plot(truth[:, 0], truth[:, 1], "k--", label="ground truth")
    axes[1].plot(truth[:, 0], truth[:, 2], "k--", label="ground truth")
    for axis, ylabel in zip(axes, ("y [m]", "z [m]")):
        axis.set_xlabel("x [m]"); axis.set_ylabel(ylabel); axis.grid(alpha=.25); axis.axis("equal")
    axes[0].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(output / "trajectory_comparison.png", dpi=160); plt.close(figure)

    names = ("ate_se3_rmse_m", "end_to_end_latency_ms_p95", "vins_cpu_percent_mean", "vins_rss_mib_max")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, name in zip(axes.flat, names):
        values = [[row[name] for row in rows if row["mode"] == mode] for mode in ("mono", "stereo")]
        axis.boxplot(values, labels=("mono", "stereo")); axis.set_title(name); axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(output / "metric_distributions.png", dpi=160); plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for mode in ("mono", "stereo"):
        for run_dir in sorted((output / mode).glob("run_*")):
            resource_rows = dict_rows(run_dir / "resources.csv")
            groups = defaultdict(list)
            for row in resource_rows:
                if row["group"] in ("feature_tracker", "vins_estimator", "roscore"):
                    groups[row["group"]].append(row)
            count = min((len(value) for value in groups.values()), default=0)
            elapsed, cpu, rss = [], [], []
            for index in range(count):
                batch = [groups[name][index] for name in ("feature_tracker", "vins_estimator", "roscore")]
                elapsed.append(float(batch[0]["elapsed_s"]))
                cpu.append(sum(float(row["cpu_percent"]) for row in batch))
                rss.append(sum(float(row["rss_bytes"]) for row in batch) / 1048576.0)
            label = f"{mode}/{run_dir.name}"
            axes[0].plot(elapsed, cpu, alpha=.65, label=label)
            axes[1].plot(elapsed, rss, alpha=.65, label=label)
    axes[0].set_ylabel("VINS CPU [%]"); axes[1].set_ylabel("VINS RSS [MiB]")
    axes[1].set_xlabel("elapsed [s]")
    for axis in axes: axis.grid(alpha=.25)
    axes[0].legend(fontsize=7, ncol=2)
    figure.tight_layout(); figure.savefig(output / "resource_timeseries.png", dpi=160); plt.close(figure)

    timing_names = ("feature_tracking_ms", "stereo_matching_ms", "optimization_ms",
                    "marginalization_ms", "backend_total_ms", "end_to_end_latency_ms")
    values = {mode: {name: [] for name in timing_names} for mode in ("mono", "stereo")}
    for mode in ("mono", "stereo"):
        for run_dir in sorted((output / mode).glob("run_*")):
            front = dict_rows(run_dir / "feature_timing.csv")
            back = dict_rows(run_dir / "backend_timing.csv")
            published = {row["timestamp_ns"]: row for row in front if row["estimator_published"] == "1"}
            backend = {row["timestamp_ns"]: row for row in back}
            for name in timing_names[:2]: values[mode][name] += finite(row[name] for row in front)
            for name in timing_names[2:5]: values[mode][name] += finite(row[name] for row in back)
            values[mode]["end_to_end_latency_ms"] += [
                (float(backend[stamp]["backend_finish_monotonic_ns"]) -
                 float(published[stamp]["frame_start_monotonic_ns"])) / 1e6
                for stamp in published.keys() & backend.keys()]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, name in zip(axes.flat, timing_names):
        axis.boxplot([values["mono"][name], values["stereo"][name]], labels=("mono", "stereo"),
                     showfliers=False)
        axis.set_title(name); axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(output / "timing_distributions.png", dpi=160); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    dataset = (args.dataset or repo / "datasets/MH_03_medium").resolve()
    output = (args.output or repo / "outputs/euroc_eval/MH_03_medium").resolve()
    rows, trajectories = [], []
    for mode in ("mono", "stereo"):
        for run_dir in sorted((output / mode).glob("run_*")):
            metrics, trajectory = evaluate_run(dataset, mode, run_dir)
            rows.append(metrics)
            trajectories.append((mode, run_dir.name, *trajectory))
    if not rows or {row["mode"] for row in rows} != {"mono", "stereo"}:
        raise SystemExit("both mono and stereo successful runs are required")
    write_csv(output / "run_metrics.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    write_comparison(output / "comparison.md", rows)
    plot_results(output, rows, trajectories)
    print("wrote benchmark report to", output)


if __name__ == "__main__":
    main()
