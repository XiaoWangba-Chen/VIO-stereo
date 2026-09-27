#!/usr/bin/env python3
import argparse
import csv
import heapq
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu


def read_camera(root, camera):
    path = root / "mav0" / camera / "data.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append((int(row[0]), row[1]))
    return rows


def pair_cameras(left_rows, right_rows):
    """Return exact timestamp pairs and deterministic unmatched counters."""
    pairs = []
    left_only = 0
    right_only = 0
    left_index = 0
    right_index = 0
    while left_index < len(left_rows) and right_index < len(right_rows):
        left_stamp, left_name = left_rows[left_index]
        right_stamp, right_name = right_rows[right_index]
        if left_stamp == right_stamp:
            pairs.append((left_stamp, left_name, right_name))
            left_index += 1
            right_index += 1
        elif left_stamp < right_stamp:
            left_only += 1
            left_index += 1
        else:
            right_only += 1
            right_index += 1
    left_only += len(left_rows) - left_index
    right_only += len(right_rows) - right_index
    return pairs, left_only, right_only


def read_depth(root):
    path = root / "mav0/depth0/data.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append((int(row[0]), row[1]))
    return rows


def read_imu(root):
    path = root / "mav0/imu0/data.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append((int(row[0]), [float(value) for value in row[1:7]]))
    return rows


def ensure_strictly_increasing(rows, stream_name):
    for previous, current in zip(rows, rows[1:]):
        if current[0] <= previous[0]:
            raise SystemExit(f"{stream_name} 时间戳不是严格递增: {previous[0]} -> {current[0]}")


def euroc_rectification_maps(root):
    sensors = []
    for camera in ("cam0", "cam1"):
        with (root / "mav0" / camera / "sensor.yaml").open() as stream:
            sensors.append(yaml.safe_load(stream))

    transforms = [np.asarray(sensor["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
                  for sensor in sensors]
    intrinsics = []
    distortions = []
    for sensor in sensors:
        fx, fy, cx, cy = sensor["intrinsics"]
        intrinsics.append(np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]))
        distortions.append(np.asarray(sensor["distortion_coefficients"], dtype=np.float64))

    rotation_01 = transforms[1][:3, :3].T @ transforms[0][:3, :3]
    translation_01 = transforms[1][:3, :3].T @ (transforms[0][:3, 3] - transforms[1][:3, 3])
    size = tuple(int(value) for value in sensors[0]["resolution"])
    rect0, rect1, proj0, proj1, _, _, _ = cv2.stereoRectify(
        intrinsics[0], distortions[0], intrinsics[1], distortions[1], size,
        rotation_01, translation_01, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map0 = cv2.initUndistortRectifyMap(
        intrinsics[0], distortions[0], rect0, proj0[:3, :3], size, cv2.CV_16SC2)
    map1 = cv2.initUndistortRectifyMap(
        intrinsics[1], distortions[1], rect1, proj1[:3, :3], size, cv2.CV_16SC2)
    return map0, map1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--stereo", action="store_true", help="publish exact cam0/cam1 pairs")
    parser.add_argument("--rectify-euroc", action="store_true",
                        help="undistort and horizontally rectify EuRoC stereo images")
    parser.add_argument("--left-topic", default="/cam0/image_raw")
    parser.add_argument("--right-topic", default="/cam1/image_raw")
    parser.add_argument("--imu-topic", default="/imu0")
    parser.add_argument("--validate-only", action="store_true", help="validate CSV pairing without ROS playback")
    args = parser.parse_args(rospy.myargv()[1:])

    if args.rate <= 0:
        raise SystemExit("--rate must be greater than zero")
    if args.rectify_euroc and not args.stereo:
        raise SystemExit("--rectify-euroc requires --stereo")

    left_rows = read_camera(args.dataset, "cam0")
    right_rows = read_camera(args.dataset, "cam1") if args.stereo else []
    depths = read_depth(args.dataset)
    imus = read_imu(args.dataset)
    if not left_rows or not imus:
        raise SystemExit("数据集中缺少 cam0 或 imu0 数据")
    ensure_strictly_increasing(left_rows, "cam0")
    ensure_strictly_increasing(imus, "imu0")
    if right_rows:
        ensure_strictly_increasing(right_rows, "cam1")
    if depths:
        ensure_strictly_increasing(depths, "depth0")

    pairs = []
    left_only = 0
    right_only = 0
    if args.stereo:
        if not right_rows:
            raise SystemExit("双目模式下数据集中缺少 cam1 数据")
        pairs, left_only, right_only = pair_cameras(left_rows, right_rows)
        if not pairs:
            raise SystemExit("cam0 与 cam1 没有时间戳完全相同的图像对")

    summary = {
        "mode": "stereo" if args.stereo else "mono",
        "left_input": len(left_rows),
        "right_input": len(right_rows),
        "pairs": len(pairs),
        "left_only": left_only,
        "right_only": right_only,
        "imu_input": len(imus),
        "depth_input": len(depths),
    }
    if args.validate_only:
        print(json.dumps(summary, sort_keys=True))
        return

    rectification_maps = euroc_rectification_maps(args.dataset) if args.rectify_euroc else None

    rospy.init_node("euroc_player")
    left_pub = rospy.Publisher(args.left_topic, Image, queue_size=10)
    right_pub = rospy.Publisher(args.right_topic, Image, queue_size=10) if args.stereo else None
    depth_pub = rospy.Publisher("/depth0/image_raw", Image, queue_size=10)
    imu_pub = rospy.Publisher(args.imu_topic, Imu, queue_size=2000)
    bridge = CvBridge()
    time.sleep(3.0)

    events = []
    camera_events = pairs if args.stereo else [(stamp, filename, None) for stamp, filename in left_rows]
    for stamp, left_filename, right_filename in camera_events:
        events.append((stamp, 1, (left_filename, right_filename)))
    for stamp, values in imus:
        events.append((stamp, 0, values))
    for stamp, filename in depths:
        events.append((stamp, 2, filename))
    heapq.heapify(events)

    published_left = 0
    published_right = 0
    published_imu = 0
    image_read_failures = 0
    first_stamp = events[0][0]
    wall_start = time.monotonic()
    try:
        while events and not rospy.is_shutdown():
            stamp, kind, payload = heapq.heappop(events)
            target = wall_start + (stamp - first_stamp) / 1e9 / args.rate
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            ros_stamp = rospy.Time(stamp // 1_000_000_000, stamp % 1_000_000_000)
            if kind == 0:
                wx, wy, wz, ax, ay, az = payload
                msg = Imu()
                msg.header.stamp = ros_stamp
                msg.header.frame_id = "camera_imu_optical_frame"
                msg.orientation_covariance[0] = -1.0
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = wx, wy, wz
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = ax, ay, az
                imu_pub.publish(msg)
                published_imu += 1
            elif kind == 1:
                left_filename, right_filename = payload
                left = cv2.imread(str(args.dataset / "mav0/cam0/data" / left_filename), cv2.IMREAD_GRAYSCALE)
                right = None
                if right_filename is not None:
                    right = cv2.imread(str(args.dataset / "mav0/cam1/data" / right_filename), cv2.IMREAD_GRAYSCALE)
                if left is None or (right_filename is not None and right is None):
                    image_read_failures += 1
                    raise RuntimeError(f"读取双目图像失败: {left_filename}, {right_filename}")
                if rectification_maps is not None:
                    left = cv2.remap(left, *rectification_maps[0], interpolation=cv2.INTER_LINEAR)
                    right = cv2.remap(right, *rectification_maps[1], interpolation=cv2.INTER_LINEAR)

                left_msg = bridge.cv2_to_imgmsg(left, encoding="mono8")
                left_msg.header.stamp = ros_stamp
                left_msg.header.frame_id = "camera_infra1_optical_frame"
                left_pub.publish(left_msg)
                published_left += 1
                if right is not None:
                    right_msg = bridge.cv2_to_imgmsg(right, encoding="mono8")
                    right_msg.header.stamp = ros_stamp
                    right_msg.header.frame_id = "camera_infra2_optical_frame"
                    right_pub.publish(right_msg)
                    published_right += 1
            else:
                depth = cv2.imread(str(args.dataset / "mav0/depth0/data" / payload), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    image_read_failures += 1
                    raise RuntimeError(f"读取深度图失败: {payload}")
                msg = bridge.cv2_to_imgmsg(depth, encoding="16UC1")
                msg.header.stamp = ros_stamp
                msg.header.frame_id = "camera_depth_optical_frame"
                depth_pub.publish(msg)
    finally:
        summary.update({
            "left_published": published_left,
            "right_published": published_right,
            "imu_published": published_imu,
            "image_read_failures": image_read_failures,
        })
        rospy.loginfo("Dataset playback summary: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
