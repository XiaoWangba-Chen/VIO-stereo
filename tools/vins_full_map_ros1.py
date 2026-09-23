#!/usr/bin/env python3
import argparse
import os
import threading
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud, PointCloud2


class FullMapAccumulator:
    def __init__(self, output_path, publish_hz, save_interval):
        self.output_path = output_path
        self.lock = threading.Lock()
        self.history_chunks = []
        self.history_count = 0
        self.current = np.empty((0, 3), dtype=np.float32)
        self.last_header = None
        self.publisher = rospy.Publisher(
            "/vins_estimator/full_map", PointCloud2, queue_size=1, latch=True)
        self.history_subscriber = rospy.Subscriber(
            "/vins_estimator/history_cloud", PointCloud, self.on_history, queue_size=1000)
        self.current_subscriber = rospy.Subscriber(
            "/vins_estimator/point_cloud", PointCloud, self.on_current, queue_size=5)
        self.publish_timer = rospy.Timer(rospy.Duration(1.0 / publish_hz), self.publish)
        self.save_timer = rospy.Timer(rospy.Duration(save_interval), self.save)
        rospy.on_shutdown(self.save)

    @staticmethod
    def points(message):
        if not message.points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray([(point.x, point.y, point.z) for point in message.points],
                          dtype=np.float32)

    def on_history(self, message):
        points = self.points(message)
        with self.lock:
            if len(points):
                # History points are appended without voxel filtering or a point-count cap.
                self.history_chunks.append(points)
                self.history_count += len(points)
            self.last_header = message.header

    def on_current(self, message):
        with self.lock:
            self.current = self.points(message)
            self.last_header = message.header

    def snapshot(self):
        with self.lock:
            chunks = list(self.history_chunks)
            current = self.current.copy()
            header = self.last_header
        if current.size:
            chunks.append(current)
        points = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.float32)
        return points, header

    def publish(self, _event=None):
        points, header = self.snapshot()
        if header is None:
            return
        header.frame_id = "world"
        self.publisher.publish(point_cloud2.create_cloud_xyz32(header, points))

    def save(self, _event=None):
        points, _ = self.snapshot()
        if not len(points):
            return
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {len(points)}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(points)}\n"
            "DATA binary\n"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            stream.write(header.encode("ascii"))
            stream.write(np.ascontiguousarray(points, dtype=np.float32).tobytes())
        os.replace(temporary, self.output_path)
        rospy.loginfo("full VINS map saved: %d points -> %s", len(points), self.output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish-hz", type=float, default=2.0)
    parser.add_argument("--save-interval", type=float, default=10.0)
    args = parser.parse_args(rospy.myargv()[1:])
    if args.publish_hz <= 0 or args.save_interval <= 0:
        raise SystemExit("publish and save intervals must be positive")
    rospy.init_node("vins_full_map_accumulator")
    FullMapAccumulator(args.output, args.publish_hz, args.save_interval)
    rospy.spin()


if __name__ == "__main__":
    main()
