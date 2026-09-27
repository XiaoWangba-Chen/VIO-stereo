# EuRoC mono/stereo benchmark

This benchmark compares the repository's monocular VIO and sliding-window stereo VIO on the
same `MH_03_medium` camera/IMU stream. Loop closure, RViz, image display, and image saving stay
disabled. The default experiment performs three successful runs per mode in the order
`mono, stereo, stereo, mono, mono, stereo`; failed attempts are retained and retried.

## Prerequisites

Build the catkin workspace, then source ROS1 and the workspace in the current shell. Verify the
environment without starting ROS processes:

```bash
cd /path/to/VINS-Mono
python3 tools/run_euroc_benchmark.py --validate-only
```

The check requires `roscore`, `rosrun`, the `feature_tracker` and `vins_estimator` ROS packages,
and all four required MH_03 data streams.

## Full experiment

```bash
python3 tools/run_euroc_benchmark.py
```

The default output is `outputs/euroc_eval/MH_03_medium`. Each successful run contains the exact
configuration, trajectory, node logs, `feature_timing.csv`, `backend_timing.csv`,
`resources.csv`, and status metadata. Failed attempts are placed below each mode's
`failed_attempts` directory and are never included in successful-run statistics.

Regenerate reports without rerunning VINS:

```bash
python3 tools/evaluate_euroc_benchmark.py
```

The report consists of `run_metrics.csv`, `summary.csv`, `comparison.md`, aligned per-run
trajectories, trajectory/metric/timing plots, and CPU/RAM time-series plots. SE(3)-aligned ATE is
the primary accuracy result; Sim(3) ATE and scale are diagnostics. RPE uses one-second relative
poses and reports translation in metres and rotation in radians.

## Short smoke run

For a true short smoke test, copy or symlink a time-consistent EuRoC subset containing cam0,
cam1, IMU, and ground truth, then pass it explicitly:

```bash
python3 tools/run_euroc_benchmark.py --dataset /path/to/subset --runs 1
```

Do not compare resource or accuracy numbers from a subset with the full-sequence report.

## Interpretation

`vins_*` resource fields combine `feature_tracker`, `vins_estimator`, and the isolated run's ROS
core; the player is reported separately. CPU percentage can exceed 100% on a multicore system.
The integrity gate requires at least 98% trajectory coverage, no estimator failure/reboot, no
published feature loss, and complete stereo pairing. The real-time reference is end-to-end P95
at or below 100 ms; it is reported independently of accuracy.
