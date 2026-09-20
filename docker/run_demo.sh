#!/usr/bin/env bash
set -euo pipefail

docker run --rm -it \
  --name vins-mono-demo \
  --net=host \
  vins-mono:noetic-demo \
  bash -lc 'mkdir -p /home/shaozu/output && \
    source /root/catkin_ws/devel/setup.bash && \
    roslaunch vins_estimator simulation.launch'
