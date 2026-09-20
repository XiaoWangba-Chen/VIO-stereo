#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPICS = {
    "/camera/camera/infra1/image_rect_raw": "cam0",
    "/camera/camera/infra2/image_rect_raw": "cam1",
    "/camera/camera/depth/image_rect_raw": "depth0",
}
INFO_TOPICS = {
    "/camera/camera/infra1/camera_info": "cam0",
    "/camera/camera/infra2/camera_info": "cam1",
    "/camera/camera/depth/camera_info": "depth0",
}
IMU_TOPIC = "/camera/camera/imu"
GYRO_TOPIC = "/camera/camera/gyro/sample"
ACCEL_TOPIC = "/camera/camera/accel/sample"


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def image_array(msg):
    enc = msg.encoding.lower()
    if enc in ("mono8", "8uc1"):
        dtype = np.uint8
    elif enc in ("16uc1", "mono16"):
        dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
    else:
        raise ValueError(f"不支持的图像编码: {msg.encoding}")
    row_values = msg.step // np.dtype(dtype).itemsize
    arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_values)
    return np.ascontiguousarray(arr[:, : msg.width])


def quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x*x + y*y + z*z + w*w
    if n == 0:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])


def transform_matrix(t):
    out = np.eye(4)
    out[:3, :3] = quat_matrix(t.rotation)
    out[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return out


def find_transform(transforms, target, source):
    graph = defaultdict(list)
    for parent, child, matrix in transforms:
        graph[child].append((parent, matrix))
        graph[parent].append((child, np.linalg.inv(matrix)))
    queue = deque([(source, np.eye(4))])
    visited = {source}
    while queue:
        frame, t_frame_source = queue.popleft()
        if frame == target:
            return t_frame_source
        for next_frame, t_next_frame in graph[frame]:
            if next_frame not in visited:
                visited.add(next_frame)
                queue.append((next_frame, t_next_frame @ t_frame_source))
    return None


def matrix_yaml(matrix):
    return {"rows": 4, "cols": 4, "data": matrix.reshape(-1).tolist()}


def main():
    parser = argparse.ArgumentParser(description="把 D435i ROS2 bag 导出为 EuRoC 风格数据集")
    parser.add_argument("sequence_dir", type=Path)
    args = parser.parse_args()
    sequence = args.sequence_dir.resolve()
    bag_dir = sequence / "rosbag2"
    if not (bag_dir / "metadata.yaml").exists():
        raise SystemExit(f"找不到 ROS2 bag: {bag_dir}")

    mav0 = sequence / "mav0"
    for sensor in ("cam0", "cam1", "depth0", "imu0"):
        (mav0 / sensor).mkdir(parents=True, exist_ok=True)
    for sensor in ("cam0", "cam1", "depth0"):
        (mav0 / sensor / "data").mkdir(exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    msg_types = {name: get_message(kind) for name, kind in topic_types.items()}

    image_rows = defaultdict(list)
    imu_rows, gyro_rows, accel_rows = [], [], []
    camera_info = {}
    frames = {}
    transforms = []
    counts = defaultdict(int)

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in msg_types:
            continue
        msg = deserialize_message(data, msg_types[topic])
        if topic in TOPICS:
            sensor = TOPICS[topic]
            ns = stamp_ns(msg)
            filename = f"{ns}.png"
            path = mav0 / sensor / "data" / filename
            if not cv2.imwrite(str(path), image_array(msg)):
                raise RuntimeError(f"图像写入失败: {path}")
            image_rows[sensor].append((ns, filename))
            frames[sensor] = msg.header.frame_id
            counts[sensor] += 1
        elif topic in INFO_TOPICS and INFO_TOPICS[topic] not in camera_info:
            camera_info[INFO_TOPICS[topic]] = msg
            frames[INFO_TOPICS[topic]] = msg.header.frame_id
        elif topic == IMU_TOPIC:
            ns = stamp_ns(msg)
            imu_rows.append((ns, msg.angular_velocity.x, msg.angular_velocity.y,
                             msg.angular_velocity.z, msg.linear_acceleration.x,
                             msg.linear_acceleration.y, msg.linear_acceleration.z))
            frames["imu0"] = msg.header.frame_id
            counts["imu0"] += 1
        elif topic == GYRO_TOPIC:
            gyro_rows.append((stamp_ns(msg), msg.angular_velocity.x,
                              msg.angular_velocity.y, msg.angular_velocity.z))
        elif topic == ACCEL_TOPIC:
            accel_rows.append((stamp_ns(msg), msg.linear_acceleration.x,
                               msg.linear_acceleration.y, msg.linear_acceleration.z))
        elif topic == "/tf_static":
            for item in msg.transforms:
                transforms.append((item.header.frame_id, item.child_frame_id,
                                   transform_matrix(item.transform)))

    for sensor, rows in image_rows.items():
        with (mav0 / sensor / "data.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["#timestamp [ns]", "filename"])
            writer.writerows(sorted(set(rows)))

    with (mav0 / "imu0" / "data.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]",
                         "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]",
                         "a_RS_S_z [m s^-2]"])
        writer.writerows(sorted(imu_rows))
    for name, rows, header in (
        ("gyro_raw.csv", gyro_rows, ["#timestamp [ns]", "wx", "wy", "wz"]),
        ("accel_raw.csv", accel_rows, ["#timestamp [ns]", "ax", "ay", "az"]),
    ):
        with (mav0 / "imu0" / name).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sorted(rows))

    imu_frame = frames.get("imu0", "camera_imu_optical_frame")
    for sensor in ("cam0", "cam1", "depth0"):
        info = camera_info.get(sensor)
        if info is None:
            continue
        t_bs = find_transform(transforms, imu_frame, frames[sensor])
        config = {
            "sensor_type": "camera" if sensor != "depth0" else "depth_camera",
            "comment": "D435i rectified infrared/depth stream; factory TF extrinsic",
            "T_BS": matrix_yaml(t_bs) if t_bs is not None else None,
            "rate_hz": 30,
            "resolution": [int(info.width), int(info.height)],
            "camera_model": "pinhole",
            "intrinsics": [float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5])],
            "distortion_model": str(info.distortion_model),
            "distortion_coefficients": [float(x) for x in info.d],
        }
        with (mav0 / sensor / "sensor.yaml").open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    imu_config = {
        "sensor_type": "imu",
        "comment": "D435i combined IMU; noise parameters must be calibrated before quantitative use",
        "T_BS": matrix_yaml(np.eye(4)),
        "rate_hz": 400,
    }
    with (mav0 / "imu0" / "sensor.yaml").open("w") as f:
        yaml.safe_dump(imu_config, f, sort_keys=False)

    summary = {"sequence": sequence.name, "counts": dict(counts), "frames": frames}
    with (sequence / "export_summary.yaml").open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()

