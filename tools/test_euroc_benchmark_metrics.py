#!/usr/bin/env python3
import importlib.util
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).with_name("euroc_benchmark_metrics.py")
SPEC = importlib.util.spec_from_file_location("benchmark_metrics", str(MODULE))
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "benchmark_runner", str(Path(__file__).with_name("run_euroc_benchmark.py")))
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


class BenchmarkMetricsTest(unittest.TestCase):
    def test_umeyama_recovers_rigid_transform(self):
        source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
        rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        target = (rotation @ source.T).T + [4, 5, 6]
        aligned, recovered, _, scale = METRICS.umeyama(source, target)
        np.testing.assert_allclose(aligned, target, atol=1e-10)
        np.testing.assert_allclose(recovered, rotation, atol=1e-10)
        self.assertAlmostEqual(scale, 1.0)

    def test_rotation_rpe(self):
        times = np.asarray([0.0, 1.0, 2.0])
        positions = np.zeros((3, 3))
        identity = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
        _, errors = METRICS.relative_pose_errors(times, positions, identity,
                                                  positions, identity)
        np.testing.assert_allclose(errors, 0.0)

    def test_percentiles_ignore_nan(self):
        self.assertEqual(METRICS.percentile([1, 2, float("nan"), 3], 50), 2.0)

    def test_quaternion_interpolation(self):
        result = METRICS.interpolate_quaternions(
            [0.0, 2.0], [[1, 0, 0, 0], [0, 0, 0, 1]], [1.0])[0]
        self.assertAlmostEqual(abs(result[0]), math.sqrt(0.5), places=6)
        self.assertAlmostEqual(abs(result[3]), math.sqrt(0.5), places=6)

    def test_dropped_frames_never_negative(self):
        self.assertEqual(METRICS.dropped_frames(100, 97), 3)
        self.assertEqual(METRICS.dropped_frames(100, 101), 0)

    def test_process_tree_sampler_reads_current_process(self):
        self.assertIn(os.getpid(), RUNNER.descendants(os.getpid()))
        ticks, rss = RUNNER.process_stats(os.getpid())
        self.assertGreaterEqual(ticks, 0)
        self.assertGreater(rss, 0)

    def test_materialized_config_preserves_absolute_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cam.yaml").write_text("calibration")
            source = root / "source.yaml"
            source.write_text('output_path: "/tmp/old/"\ncam0_calib: "cam.yaml"\n')
            destination = root / "run/config.yaml"
            destination.parent.mkdir()
            RUNNER.materialize_config(source, destination, destination.parent)
            text = destination.read_text()
            self.assertIn(str((root / "cam.yaml").resolve()), text)
            self.assertIn(str(destination.parent.resolve()), text)


if __name__ == "__main__":
    unittest.main()
