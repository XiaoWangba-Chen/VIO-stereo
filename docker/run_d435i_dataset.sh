#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "用法: $0 序列名（例如 room_02）" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sequence="$1"
dataset_dir="${repo_dir}/datasets/${sequence}"

if [[ ! -f "${dataset_dir}/mav0/cam0/data.csv" || ! -f "${dataset_dir}/mav0/imu0/data.csv" ]]; then
  echo "数据集未导出或不完整: ${dataset_dir}" >&2
  exit 1
fi
if [[ -z "${DISPLAY:-}" ]]; then
  echo "请在图形桌面的终端中运行。" >&2
  exit 1
fi

xhost +SI:localuser:root >/dev/null
cleanup() { xhost -SI:localuser:root >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker run --rm -it \
  --name vins-mono-d435i \
  --net=host \
  --env DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
  --volume "${dataset_dir}:/dataset:ro" \
  --volume "${repo_dir}/tools/play_euroc_ros1.py:/play_euroc_ros1.py:ro" \
  --volume "${repo_dir}/config/d435i/d435i_ir_config.yaml:/d435i_ir_config.yaml:ro" \
  vins-mono:noetic-demo \
  bash -lc '
    mkdir -p /tmp/vins_output/pose_graph
    source /root/catkin_ws/devel/setup.bash
    roslaunch vins_estimator realsense_color.launch config_path:=/d435i_ir_config.yaml &
    launch_pid=$!
    trap "kill -INT ${launch_pid} 2>/dev/null || true" EXIT INT TERM
    sleep 5
    python3 /play_euroc_ros1.py /dataset &
    player_pid=$!
    rviz -d /root/catkin_ws/src/VINS-Mono/config/vins_rviz_config.rviz
    kill -INT ${player_pid} 2>/dev/null || true
  '
