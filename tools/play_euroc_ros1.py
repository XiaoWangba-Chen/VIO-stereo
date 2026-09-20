#!/usr/bin/env python3
import argparse
import csv
import heapq
import time
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu


def read_camera(root):
    rows = []
    with (root / "mav0/cam0/data.csv").open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append((int(row[0]), row[1]))
    return rows


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
    rows = []
    with (root / "mav0/imu0/data.csv").open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append((int(row[0]), [float(value) for value in row[1:7]]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rate", type=float, default=1.0)
    args = parser.parse_args(rospy.myargv()[1:])

    cameras = read_camera(args.dataset)
    depths = read_depth(args.dataset)
    imus = read_imu(args.dataset)
    if not cameras or not imus:
        raise SystemExit("数据集中缺少 cam0 或 imu0 数据")

    rospy.init_node("euroc_d435i_player")
    image_pub = rospy.Publisher("/d435i/infra1/image_raw", Image, queue_size=10)
    depth_pub = rospy.Publisher("/d435i/depth/image_raw", Image, queue_size=10)
    imu_pub = rospy.Publisher("/d435i/imu", Imu, queue_size=2000)
    bridge = CvBridge()
    time.sleep(3.0)

    events = []
    for stamp, filename in cameras:
        events.append((stamp, 1, filename))
    for stamp, values in imus:
        events.append((stamp, 0, values))
    for stamp, filename in depths:
        events.append((stamp, 2, filename))
    heapq.heapify(events)

    first_stamp = events[0][0]
    wall_start = time.monotonic()
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
        elif kind == 1:
            image = cv2.imread(str(args.dataset / "mav0/cam0/data" / payload), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"读取图像失败: {payload}")
            msg = bridge.cv2_to_imgmsg(image, encoding="mono8")
            msg.header.stamp = ros_stamp
            msg.header.frame_id = "camera_infra1_optical_frame"
            image_pub.publish(msg)
        else:
            depth = cv2.imread(str(args.dataset / "mav0/depth0/data" / payload), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"读取深度图失败: {payload}")
            msg = bridge.cv2_to_imgmsg(depth, encoding="16UC1")
            msg.header.stamp = ros_stamp
            msg.header.frame_id = "camera_depth_optical_frame"
            depth_pub.publish(msg)

    rospy.loginfo("Dataset playback finished")


if __name__ == "__main__":
    main()
