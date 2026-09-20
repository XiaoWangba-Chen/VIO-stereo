#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "用法: $0 序列名（例如 room_01）" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sequence="$1"
output_dir="${repo_dir}/datasets/${sequence}"
bag_dir="${output_dir}/rosbag2"

if [[ -e "$output_dir" ]]; then
  echo "目标已存在，避免覆盖: $output_dir" >&2
  exit 1
fi

mkdir -p "$output_dir"
source /opt/ros/humble/setup.bash
set -u

camera_pid=""
bag_pid=""
cleanup() {
  trap - INT TERM EXIT
  if [[ -n "$bag_pid" ]] && kill -0 "$bag_pid" 2>/dev/null; then
    kill -INT "$bag_pid"
    wait "$bag_pid" || true
  fi
  if [[ -n "$camera_pid" ]] && kill -0 "$camera_pid" 2>/dev/null; then
    kill -INT "$camera_pid"
    wait "$camera_pid" || true
  fi
  echo "录制已保存到: $output_dir"
}
trap cleanup INT TERM EXIT

rs-enumerate-devices -s > "${output_dir}/device.txt"

ros2 launch realsense2_camera rs_launch.py \
  serial_no:="'939622074967'" \
  enable_color:=false \
  enable_depth:=true \
  enable_infra1:=true \
  enable_infra2:=true \
  depth_module.depth_profile:=640,480,30 \
  depth_module.infra_profile:=640,480,30 \
  enable_gyro:=true \
  enable_accel:=true \
  gyro_fps:=400 \
  accel_fps:=250 \
  unite_imu_method:=2 \
  enable_sync:=true \
  initial_reset:=false \
  > "${output_dir}/camera.log" 2>&1 &
camera_pid=$!

echo "等待 D435i 数据流启动……"
sleep 8

if ! kill -0 "$camera_pid" 2>/dev/null; then
  echo "D435i 驱动启动失败，请查看 ${output_dir}/camera.log" >&2
  exit 1
fi

if ! timeout 8s ros2 topic echo --once /camera/camera/imu >/dev/null 2>&1; then
  echo "没有收到合并 IMU 数据，请重新插拔相机后再试；详情见 ${output_dir}/camera.log" >&2
  exit 1
fi

ros2 param set /camera/camera depth_module.emitter_enabled 0 >/dev/null
emitter_state="$(ros2 param get /camera/camera depth_module.emitter_enabled)"
echo "主动红外状态: ${emitter_state}"

ros2 bag record -s sqlite3 -o "$bag_dir" --max-cache-size 268435456 \
  /camera/camera/infra1/image_rect_raw \
  /camera/camera/infra1/camera_info \
  /camera/camera/infra2/image_rect_raw \
  /camera/camera/infra2/camera_info \
  /camera/camera/depth/image_rect_raw \
  /camera/camera/depth/camera_info \
  /camera/camera/gyro/sample \
  /camera/camera/accel/sample \
  /camera/camera/imu \
  /camera/camera/gyro/imu_info \
  /camera/camera/accel/imu_info \
  /camera/camera/extrinsics/depth_to_accel \
  /camera/camera/extrinsics/depth_to_gyro \
  /camera/camera/extrinsics/depth_to_infra1 \
  /camera/camera/extrinsics/depth_to_infra2 \
  /tf /tf_static &
bag_pid=$!

echo
echo "正在录制 ${sequence}：IR1 + IR2 + Depth + IMU"
echo "请移动相机；结束时按 Ctrl+C。"
wait "$bag_pid"
