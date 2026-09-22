#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "用法: $0 序列名 [--headless] [--mode a1|b0|b1] [--rate 正数] [--image 镜像] [--label 标签] [--validate-depth]" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sequence="$1"
dataset_dir="${repo_dir}/datasets/${sequence}"
mode=b0
headless=0
playback_rate=1.0
validate_depth=0
docker_image="${VINS_DOCKER_IMAGE:-vins-mono:noetic-demo}"
run_label_override="${VINS_RUN_LABEL:-}"
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless)
      headless=1
      shift
      ;;
    --mode)
      if [[ $# -lt 2 || ( "$2" != "a1" && "$2" != "b0" && "$2" != "b1" ) ]]; then
        echo "--mode 必须是 a1、b0 或 b1" >&2
        exit 2
      fi
      mode="$2"
      shift 2
      ;;
    --validate-depth)
      validate_depth=1
      shift
      ;;
    --rate)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+([.][0-9]+)?$ || "$2" == "0" || "$2" == "0.0" ]]; then
        echo "--rate 必须是正数" >&2
        exit 2
      fi
      playback_rate="$2"
      shift 2
      ;;
    --image)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "--image 需要 Docker 镜像名" >&2
        exit 2
      fi
      docker_image="$2"
      shift 2
      ;;
    --label)
      if [[ $# -lt 2 ]]; then
        echo "--label 需要实验标签" >&2
        exit 2
      fi
      run_label_override="$2"
      shift 2
      ;;
    *)
      echo "未知选项: $1" >&2
      exit 2
      ;;
  esac
done

run_label="${run_label_override:-${mode}}"
if [[ ! "${run_label}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "VINS_RUN_LABEL 只能包含字母、数字、点、下划线和连字符" >&2
  exit 2
fi
output_dir="${repo_dir}/outputs/${sequence}/${run_label}"
config_file="${repo_dir}/config/d435i/${sequence}_${mode}.yaml"
if [[ ! -f "${config_file}" ]]; then
  echo "缺少 ${mode} 实验配置: ${config_file}" >&2
  exit 1
fi
container_config="/d435i_config/${sequence}_${mode}.yaml"

if [[ ! -f "${dataset_dir}/mav0/cam0/data.csv" || ! -f "${dataset_dir}/mav0/imu0/data.csv" ]]; then
  echo "数据集未导出或不完整: ${dataset_dir}" >&2
  exit 1
fi
if [[ "${mode}" != "a1" && ! -f "${dataset_dir}/mav0/cam1/data.csv" ]]; then
  echo "${mode} 模式需要 cam1 数据: ${dataset_dir}" >&2
  exit 1
fi
if [[ "${validate_depth}" -eq 1 ]]; then
  if [[ "${mode}" != "b1" ]]; then
    echo "--validate-depth 仅支持 b1 模式" >&2
    exit 2
  fi
  if [[ ! -f "${dataset_dir}/mav0/depth0/data.csv" ]]; then
    echo "--validate-depth 需要 depth0 数据: ${dataset_dir}" >&2
    exit 1
  fi
fi
max_cnt="$(awk '$1 == "max_cnt:" {print $2; exit}' "${config_file}")"
if [[ ! "${max_cnt}" =~ ^[0-9]+$ || "${max_cnt}" -lt 2 ]]; then
  echo "无法从配置读取有效的 max_cnt: ${config_file}" >&2
  exit 1
fi
min_stereo_candidates="$((max_cnt / 2))"
mkdir -p "${output_dir}/pose_graph"
if [[ "${headless}" -eq 0 && -z "${DISPLAY:-}" ]]; then
  echo "请在图形桌面的终端中运行。" >&2
  exit 1
fi

docker_args=(--rm --name vins-mono-d435i --net=host)
if [[ "${headless}" -eq 0 ]]; then
  xhost +SI:localuser:root >/dev/null
  cleanup() { xhost -SI:localuser:root >/dev/null 2>&1 || true; }
  trap cleanup EXIT
  docker_args+=(--env DISPLAY --env QT_X11_NO_MITSHM=1 --volume /tmp/.X11-unix:/tmp/.X11-unix:ro)
fi

docker run "${docker_args[@]}" \
  --volume "${dataset_dir}:/dataset:ro" \
  --volume "${output_dir}:/tmp/vins_output" \
  --volume "${repo_dir}/tools/play_euroc_ros1.py:/play_euroc_ros1.py:ro" \
  --volume "${repo_dir}/tools/validate_stereo_depth_ros1.py:/validate_stereo_depth_ros1.py:ro" \
  --volume "${repo_dir}/config/d435i:/d435i_config:ro" \
  --env VINS_HEADLESS="${headless}" \
  --env VINS_PHASE1_MODE="${mode}" \
  --env VINS_CONFIG="${container_config}" \
  --env VINS_PLAYBACK_RATE="${playback_rate}" \
  --env VINS_VALIDATE_DEPTH="${validate_depth}" \
  --env VINS_MIN_STEREO_CANDIDATES="${min_stereo_candidates}" \
  --env ROS_LOG_DIR="/tmp/vins_output/ros_log" \
  "${docker_image}" \
  bash -lc '
    mkdir -p /tmp/vins_output/pose_graph "${ROS_LOG_DIR}"
    source /root/catkin_ws/devel/setup.bash
    roslaunch vins_estimator realsense_color.launch config_path:="${VINS_CONFIG}" &
    launch_pid=$!
    shutdown_launch() {
      kill -INT "${launch_pid}" 2>/dev/null || true
      wait "${launch_pid}" 2>/dev/null || true
    }
    trap shutdown_launch EXIT INT TERM
    sleep 5
    validator_pid=""
    if [[ "${VINS_PHASE1_MODE}" == "b1" ]]; then
      validator_args=(--output /tmp/vins_output/stereo_diagnostics.json
                      --min-candidates "${VINS_MIN_STEREO_CANDIDATES}")
      if [[ "${VINS_VALIDATE_DEPTH}" == "1" ]]; then
        validator_args+=(--expect-depth)
      fi
      python3 /validate_stereo_depth_ros1.py "${validator_args[@]}" &
      validator_pid=$!
      sleep 1
    fi
    player_args=(/dataset --rate "${VINS_PLAYBACK_RATE}")
    if [[ "${VINS_PHASE1_MODE}" != "a1" ]]; then
      player_args+=(--stereo)
    fi
    python3 /play_euroc_ros1.py "${player_args[@]}" &
    player_pid=$!
    if [[ "${VINS_HEADLESS}" == "1" ]]; then
      wait ${player_pid}
    else
      rviz -d /root/catkin_ws/src/VINS-Mono/config/vins_rviz_config.rviz
      kill -INT ${player_pid} 2>/dev/null || true
      wait ${player_pid} 2>/dev/null || true
    fi
    # 保持 B1 工作线程和诊断订阅者存活，让播放结束时的少量匹配任务排空。
    if [[ "${VINS_PHASE1_MODE}" == "b1" ]]; then
      sleep 10
    fi
    if [[ -n "${validator_pid}" ]]; then
      kill -INT "${validator_pid}" 2>/dev/null || true
      wait "${validator_pid}" 2>/dev/null || true
    fi
    sleep 1
    shutdown_launch
    trap - EXIT INT TERM
  '
