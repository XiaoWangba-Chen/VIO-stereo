#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
  echo "没有检测到图形桌面（DISPLAY 未设置），请在桌面终端中运行。" >&2
  exit 1
fi

xhost +SI:localuser:root >/dev/null
cleanup() {
  xhost -SI:localuser:root >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run --rm -it \
  --name vins-mono-demo \
  --net=host \
  --env DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
  vins-mono:noetic-demo \
  bash -lc '
    mkdir -p /home/shaozu/output
    source /root/catkin_ws/devel/setup.bash
    roslaunch vins_estimator simulation.launch &
    launch_pid=$!
    trap "kill -INT ${launch_pid} 2>/dev/null || true" EXIT INT TERM
    sleep 3
    rviz -d /root/catkin_ws/src/VINS-Mono/config/vins_rviz_config.rviz
  '
