#!/usr/bin/env python3
import threading
import argparse
from collections import deque

import numpy as np
import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header


FX = 384.4474792480469
FY = 384.4474792480469
CX = 318.49029541015625
CY = 241.03326416015625
T_BODY_DEPTH = np.array([-0.005520000122487542, 0.005100000184029343,
                         0.011739999987185001])


def rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


class Mapper:
    def __init__(self, output_path):
        self.output_path = output_path
        self.bridge = CvBridge()
        self.depths = deque(maxlen=120)
        self.voxels = {}
        self.lock = threading.Lock()
        self.frame_count = 0
        self.publisher = rospy.Publisher("/d435i/map_points", PointCloud2,
                                         queue_size=1, latch=True)
        rospy.Subscriber("/d435i/depth/image_raw", Image, self.depth_callback,
                         queue_size=30, buff_size=8 * 1024 * 1024)
        rospy.Subscriber("/vins_estimator/odometry", Odometry,
                         self.odom_callback, queue_size=100)
        rospy.on_shutdown(self.save_pcd)

    def save_pcd(self):
        with self.lock:
            points = list(self.voxels.values())
        if not points or not self.output_path:
            return
        try:
            with open(self.output_path, "w") as f:
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n")
                f.write("COUNT 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n" % len(points))
                f.write("POINTS %d\nDATA ascii\n" % len(points))
                for p in points:
                    f.write("%.5f %.5f %.5f\n" % (p[0], p[1], p[2]))
            rospy.loginfo("Saved dense map: %s (%d points)", self.output_path, len(points))
        except Exception as exc:
            rospy.logwarn("Could not save PCD: %s", exc)

    def depth_callback(self, msg):
        with self.lock:
            self.depths.append((msg.header.stamp.to_nsec(), msg))

    def odom_callback(self, odom):
        stamp = odom.header.stamp.to_nsec()
        with self.lock:
            if not self.depths:
                return
            _, depth_msg = min(self.depths, key=lambda item: abs(item[0] - stamp))
        if abs(depth_msg.header.stamp.to_nsec() - stamp) > 20_000_000:
            return
        self.frame_count += 1
        if self.frame_count % 3:
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        v, u = np.mgrid[0:depth.shape[0]:8, 0:depth.shape[1]:8]
        z = depth[::8, ::8].astype(np.float64) * 0.001
        valid = (z > 0.25) & (z < 5.0)
        z, u, v = z[valid], u[valid], v[valid]
        if z.size == 0:
            return
        points = np.column_stack(((u-CX)*z/FX, (v-CY)*z/FY, z))
        points += T_BODY_DEPTH

        pose = odom.pose.pose
        r_world_body = rotation_matrix(pose.orientation)
        t_world_body = np.array([pose.position.x, pose.position.y, pose.position.z])
        points = points @ r_world_body.T + t_world_body

        keys = np.floor(points / 0.08).astype(np.int32)
        for key, point in zip(keys, points):
            self.voxels.setdefault(tuple(key), point.astype(np.float32))
        if self.frame_count % 15 == 0:
            header = Header(stamp=odom.header.stamp, frame_id="world")
            cloud = point_cloud2.create_cloud_xyz32(header, list(self.voxels.values()))
            self.publisher.publish(cloud)
            rospy.loginfo("Dense map: %d voxels", len(self.voxels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    rospy.init_node("depth_vins_mapper")
    Mapper(args.output)
    rospy.spin()
