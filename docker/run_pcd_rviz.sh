#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "用法: $0 序列名（例如 room_02）" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pcd="${repo_dir}/outputs/$1/dense_map.pcd"
[[ -f "${pcd}" ]] || { echo "找不到点云文件: ${pcd}" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "请在图形桌面终端运行" >&2; exit 1; }

xhost +SI:localuser:root >/dev/null
cleanup() { xhost -SI:localuser:root >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run --rm -it --net=host \
  --env DISPLAY --env QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "${pcd}:/map/dense_map.pcd:ro" \
  -v "${repo_dir}/config/d435i/d435i_mapping.rviz:/mapping.rviz:ro" \
  vins-mono:noetic-demo bash -lc '
    source /root/catkin_ws/devel/setup.bash
    roscore &
    master_pid=$!
    sleep 2
    rosrun pcl_ros pcd_to_pointcloud /map/dense_map.pcd 1.0 \
      __name:=pcd_map _frame_id:=world cloud_pcd:=/d435i/map_points &
    pcd_pid=$!
    trap "kill ${pcd_pid} ${master_pid} 2>/dev/null || true" EXIT INT TERM
    rviz -d /mapping.rviz
  '
