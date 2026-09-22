#!/usr/bin/env python3
import argparse
import json
import math
import threading
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud


REJECT_NAMES = {
    0: "accepted",
    1: "flow",
    2: "border",
    3: "fb",
    4: "epipolar",
    5: "disparity",
    6: "depth_range",
}


def percentile_or_none(values, percentile):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def channels_by_name(message):
    return {channel.name: channel.values for channel in message.channels}


class StereoDepthValidator:
    def __init__(self, output_path, min_candidates, expect_depth):
        self.output_path = Path(output_path)
        self.min_candidates = min_candidates
        self.expect_depth = expect_depth
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.pending_diagnostics = OrderedDict()
        self.pending_depth = OrderedDict()
        self.finished = False

        self.diagnostic_frames = 0
        self.textured_rates = []
        self.epipolar_errors = []
        self.reject_counts = Counter()
        self.depth_frames = 0
        self.paired_depth_frames = 0
        self.accepted_for_depth = 0
        self.invalid_hardware_depth = 0
        self.out_of_range_hardware_depth = 0
        self.relative_errors = []

        rospy.Subscriber("/feature_tracker/stereo_feature", PointCloud,
                         self.diagnostic_callback, queue_size=100)
        if expect_depth:
            rospy.Subscriber("/d435i/depth/image_raw", Image,
                             self.depth_callback, queue_size=100,
                             buff_size=8 * 1024 * 1024)
        rospy.on_shutdown(self.finish)

    @staticmethod
    def _trim(cache, limit=120):
        while len(cache) > limit:
            cache.popitem(last=False)

    def diagnostic_callback(self, message):
        try:
            channels = channels_by_name(message)
            required = ("left_u", "left_v", "stereo_depth",
                        "epipolar_error", "reject_code")
            if any(name not in channels for name in required):
                rospy.logerr_throttle(5.0, "stereo diagnostic message is missing required channels")
                return
            count = len(message.points)
            if any(len(channels[name]) != count for name in required):
                rospy.logerr_throttle(5.0, "stereo diagnostic channel lengths do not match")
                return

            accepted = []
            accepted_count = 0
            for index in range(count):
                code = int(round(channels["reject_code"][index]))
                self.reject_counts[REJECT_NAMES.get(code, "unknown")] += 1
                if code != 0:
                    continue
                accepted_count += 1
                epipolar = channels["epipolar_error"][index]
                if math.isfinite(epipolar):
                    self.epipolar_errors.append(float(epipolar))
                depth = channels["stereo_depth"][index]
                left_u = channels["left_u"][index]
                left_v = channels["left_v"][index]
                if all(math.isfinite(value) for value in (depth, left_u, left_v)):
                    accepted.append((float(left_u), float(left_v), float(depth)))

            self.diagnostic_frames += 1
            if count >= self.min_candidates:
                self.textured_rates.append(accepted_count / float(count))

            if self.expect_depth:
                stamp = message.header.stamp.to_nsec()
                with self.lock:
                    self.pending_diagnostics[stamp] = accepted
                    self._trim(self.pending_diagnostics)
                    self._pair_locked(stamp)
        except Exception as error:
            rospy.logerr_throttle(5.0, "failed to process stereo diagnostics: %s", error)

    def depth_callback(self, message):
        try:
            depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="16UC1")
            stamp = message.header.stamp.to_nsec()
            self.depth_frames += 1
            with self.lock:
                self.pending_depth[stamp] = depth
                self._trim(self.pending_depth)
                self._pair_locked(stamp)
        except Exception as error:
            rospy.logerr_throttle(5.0, "failed to decode hardware depth: %s", error)

    def _pair_locked(self, stamp):
        if stamp not in self.pending_diagnostics or stamp not in self.pending_depth:
            return
        matches = self.pending_diagnostics.pop(stamp)
        depth_image = self.pending_depth.pop(stamp)
        self.paired_depth_frames += 1
        height, width = depth_image.shape[:2]

        for left_u, left_v, stereo_depth in matches:
            self.accepted_for_depth += 1
            u = int(round(left_u))
            v = int(round(left_v))
            if u < 0 or v < 0 or u >= width or v >= height:
                self.invalid_hardware_depth += 1
                continue
            x0, x1 = max(0, u - 1), min(width, u + 2)
            y0, y1 = max(0, v - 1), min(height, v + 2)
            patch = depth_image[y0:y1, x0:x1].astype(np.float64) * 0.001
            valid = patch[patch > 0.0]
            if valid.size == 0:
                self.invalid_hardware_depth += 1
                continue
            hardware_depth = float(np.median(valid))
            if hardware_depth < 0.3 or hardware_depth > 8.0:
                self.out_of_range_hardware_depth += 1
                continue
            self.relative_errors.append(abs(stereo_depth - hardware_depth) / hardware_depth)

    def summary(self):
        median_relative = percentile_or_none(self.relative_errors, 50)
        p90_relative = percentile_or_none(self.relative_errors, 90)
        return {
            "diagnostic_frames": self.diagnostic_frames,
            "min_candidates_for_textured_frame": self.min_candidates,
            "textured_frames": len(self.textured_rates),
            "textured_match_rate_median": percentile_or_none(self.textured_rates, 50),
            "epipolar_error_median_px": percentile_or_none(self.epipolar_errors, 50),
            "reject_counts": dict(sorted(self.reject_counts.items())),
            "depth_expected": self.expect_depth,
            "depth_frames": self.depth_frames,
            "paired_depth_frames": self.paired_depth_frames,
            "accepted_for_depth": self.accepted_for_depth,
            "depth_comparisons": len(self.relative_errors),
            "invalid_hardware_depth": self.invalid_hardware_depth,
            "out_of_range_hardware_depth": self.out_of_range_hardware_depth,
            "absolute_relative_error_median": median_relative,
            "absolute_relative_error_p90": p90_relative,
            "depth_target_passed": (
                median_relative is not None and p90_relative is not None and
                median_relative <= 0.10 and p90_relative <= 0.25
            ) if self.expect_depth else None,
        }

    def finish(self):
        with self.lock:
            if self.finished:
                return
            self.finished = True
        summary = self.summary()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        rospy.loginfo("Stereo validation summary: %s", json.dumps(summary, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--expect-depth", action="store_true")
    args = parser.parse_args(rospy.myargv()[1:])
    if args.min_candidates < 1:
        raise SystemExit("--min-candidates must be positive")
    rospy.init_node("stereo_depth_validator")
    StereoDepthValidator(args.output, args.min_candidates, args.expect_depth)
    rospy.spin()


if __name__ == "__main__":
    main()
