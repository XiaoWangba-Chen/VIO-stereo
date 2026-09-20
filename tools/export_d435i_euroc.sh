#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "用法: $0 序列名（例如 room_01）" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
python3 "${repo_dir}/tools/export_d435i_euroc.py" "${repo_dir}/datasets/$1"
