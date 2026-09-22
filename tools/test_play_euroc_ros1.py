#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("play_euroc_ros1.py")
SPEC = importlib.util.spec_from_file_location("play_euroc_ros1", str(MODULE_PATH))
PLAYER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAYER)


class ExactCameraPairingTest(unittest.TestCase):
    def test_exact_pairs(self):
        pairs, left_only, right_only = PLAYER.pair_cameras(
            [(10, "l0"), (20, "l1"), (30, "l2")],
            [(10, "r0"), (20, "r1"), (30, "r2")],
        )
        self.assertEqual(pairs, [(10, "l0", "r0"), (20, "l1", "r1"), (30, "l2", "r2")])
        self.assertEqual((left_only, right_only), (0, 0))

    def test_missing_right_frame(self):
        pairs, left_only, right_only = PLAYER.pair_cameras(
            [(10, "l0"), (20, "l1"), (30, "l2")],
            [(10, "r0"), (30, "r2")],
        )
        self.assertEqual(pairs, [(10, "l0", "r0"), (30, "l2", "r2")])
        self.assertEqual((left_only, right_only), (1, 0))

    def test_one_nanosecond_is_not_a_pair(self):
        pairs, left_only, right_only = PLAYER.pair_cameras(
            [(10, "left")],
            [(11, "right")],
        )
        self.assertEqual(pairs, [])
        self.assertEqual((left_only, right_only), (1, 1))

    def test_rejects_duplicate_or_decreasing_timestamps(self):
        with self.assertRaises(SystemExit):
            PLAYER.ensure_strictly_increasing([(10, "a"), (10, "b")], "cam0")
        with self.assertRaises(SystemExit):
            PLAYER.ensure_strictly_increasing([(11, "a"), (10, "b")], "cam0")


if __name__ == "__main__":
    unittest.main()
