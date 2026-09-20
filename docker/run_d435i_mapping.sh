#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "用法: $0 序列名（例如 room_02）" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_dir="${repo_dir}/datasets/$1"
output_dir="${repo_dir}/outputs/$1"
mkdir -p "${output_dir}"
[[ -f "${dataset_dir}/mav0/depth0/data.csv" ]] || { echo "缺少depth0数据" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "请在图形桌面终端运行" >&2; exit 1; }

docker stop vins-mono-d435i-map >/dev/null 2>&1 || true
xhost +SI:localuser:root >/dev/null
cleanup() { xhost -SI:localuser:root >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run --rm -it \
  --name vins-mono-d435i-map --net=host \
  --env DISPLAY --env QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "${dataset_dir}:/dataset:ro" \
  -v "${repo_dir}/tools/play_euroc_ros1.py:/player.py:ro" \
  -v "${repo_dir}/tools/depth_vins_mapper_ros1.py:/mapper.py:ro" \
  -v "${output_dir}:/output" \
  -v "${repo_dir}/config/d435i/d435i_ir_config.yaml:/vins.yaml:ro" \
  -v "${repo_dir}/config/d435i/d435i_mapping.rviz:/mapping.rviz:ro" \
  vins-mono:noetic-demo bash -lc '
    mkdir -p /tmp/vins_output/pose_graph
    source /root/catkin_ws/devel/setup.bash
    roslaunch vins_estimator realsense_color.launch config_path:=/vins.yaml &
    launch_pid=$!
    trap "kill -INT ${launch_pid} 2>/dev/null || true" EXIT INT TERM
    sleep 5
    python3 /mapper.py --output /output/dense_map.pcd &
    mapper_pid=$!
    python3 /player.py /dataset &
    player_pid=$!
    rviz -d /mapping.rviz
    kill -INT ${player_pid} ${mapper_pid} 2>/dev/null || true
  '
