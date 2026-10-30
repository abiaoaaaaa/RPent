#!/usr/bin/env bash
# Restart only the robot's quad camera with the user high-quality parameters.
# This script does not store an SSH password and does not use pkill.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RPENT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_ENV="${RPENT_ROOT}/robots/luna/config/remote_ros_env.sh"

REMOTE_USER="nvidia"
REMOTE_HOST="6.6.7.100"
REMOTE_SCRIPT="/home/nvidia/start_quadcam.sh"
REMOTE_LOG="/tmp/quadcam_quality.log"
VLAHOST_URL="http://${REMOTE_HOST}:8011"
DRY_RUN=0
VERIFY_SECONDS=6
CONFIRM_STATIC=0

usage() {
  cat <<'EOF'
用法：restart_quadcam_high_quality.sh [选项]

安全地停止机器人上已有的 gmsl_quadcam 启动进程，运行高清脚本，
再从本机验证 1920x1080、JPEG quality 90、单发布者和实时帧率。

选项：
  --remote-user USER      SSH 用户（默认 nvidia）
  --remote-host HOST      机器人地址（默认 6.6.7.100）
  --start-script PATH     机器人上的高清启动脚本
  --remote-log PATH       机器人上的启动日志
  --vlahost-url URL       VLAHost 地址（默认 http://6.6.7.100:8011）
  --verify-seconds N      相机帧验证时长（默认 6）
  --confirm-static        确认机器人静止、急停可触达且无运动任务
  --dry-run               只显示步骤，不停止或启动服务
  -h, --help              显示帮助
EOF
}

die() {
  echo "错误：$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --remote-user) [[ $# -ge 2 ]] || die "$1 需要参数"; REMOTE_USER="$2"; shift 2 ;;
    --remote-host) [[ $# -ge 2 ]] || die "$1 需要参数"; REMOTE_HOST="$2"; shift 2 ;;
    --start-script) [[ $# -ge 2 ]] || die "$1 需要参数"; REMOTE_SCRIPT="$2"; shift 2 ;;
    --remote-log) [[ $# -ge 2 ]] || die "$1 需要参数"; REMOTE_LOG="$2"; shift 2 ;;
    --vlahost-url) [[ $# -ge 2 ]] || die "$1 需要参数"; VLAHOST_URL="$2"; shift 2 ;;
    --verify-seconds) [[ $# -ge 2 ]] || die "$1 需要参数"; VERIFY_SECONDS="$2"; shift 2 ;;
    --confirm-static) CONFIRM_STATIC=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知选项：$1" ;;
  esac
done

[[ "$VERIFY_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "--verify-seconds 必须是正整数"
command -v ssh >/dev/null || die "找不到 ssh"
TARGET="${REMOTE_USER}@${REMOTE_HOST}"

echo "目标机器人：${TARGET}"
echo "高清脚本：${REMOTE_SCRIPT}"
echo "操作前必须确认：机器人静止、急停可触达、没有正在执行的运动任务。"

if ((DRY_RUN)); then
  cat <<EOF
[dry-run] ssh ${TARGET}：查找并仅向匹配的 gmsl_quadcam 启动进程发送 SIGINT
[dry-run] 等待旧进程退出后运行：bash ${REMOTE_SCRIPT}
[dry-run] 机器人日志：${REMOTE_LOG}
[dry-run] 本机 ROS 验证：1920x1080 / quality 90 / 单发布者 / 实时帧率
EOF
  exit 0
fi

((CONFIRM_STATIC)) || die "执行重启前必须加 --confirm-static，确认机器人静止、急停可触达且无运动任务"
[[ -f "$ROS_ENV" ]] || die "找不到 ROS 环境脚本：$ROS_ENV"

echo "[1/3] 远端停止旧相机并启动高清配置（SSH 可能提示密码）……"
ssh -o ConnectTimeout=8 "$TARGET" bash -s -- "$REMOTE_SCRIPT" "$REMOTE_LOG" <<'REMOTE_SCRIPT_BODY'
set -Eeuo pipefail
start_script="$1"
remote_log="$2"
launch_pattern='ros2 launch gmsl_quadcam'

# 只匹配 launch 命令，不匹配 component container 或其他机器人服务。
mapfile -t launch_pids < <(pgrep -f -- "$launch_pattern" || true)
if ((${#launch_pids[@]})); then
  printf '发现相机启动进程：'
  printf ' %s' "${launch_pids[@]}"
  printf '\n'
  for pid in "${launch_pids[@]}"; do
    if [[ -d "/proc/$pid" ]]; then
      printf '向 PID %s 发送 SIGINT：' "$pid"
      tr '\0' ' ' <"/proc/$pid/cmdline" || true
      printf '\n'
      kill -INT "$pid"
    fi
  done
else
  echo '未发现已有 gmsl_quadcam 启动进程。'
fi

deadline=$((SECONDS + 15))
while :; do
  mapfile -t remaining < <(pgrep -f -- "$launch_pattern" || true)
  ((${#remaining[@]} == 0)) && break
  ((SECONDS < deadline)) || {
    printf '旧相机启动进程仍存在：'
    printf ' %s' "${remaining[@]}"
    printf '\n' >&2
    exit 20
  }
  sleep 1
done

[[ -f "$start_script" ]] || { echo "高清启动脚本不存在：$start_script" >&2; exit 21; }
echo "启动高清相机：$start_script"
nohup setsid bash "$start_script" >"$remote_log" 2>&1 </dev/null &
echo "高清启动命令已提交，日志：$remote_log"
REMOTE_SCRIPT_BODY

echo "[2/3] 等待 ROS 相机节点上线……"
# ROS Humble's setup scripts reference optional variables while being sourced.
# Keep nounset for this wrapper, but disable it during environment loading.
set +u
source "$ROS_ENV"
set -u

param_value() {
  local name="$1"
  local output
  output="$(ros2 param get /quad_csi_quickview "$name" 2>/dev/null || true)"
  printf '%s\n' "$output" | sed -n 's/.*value is: //p'
}

width=''; height=''; quality=''
for _ in $(seq 1 20); do
  width="$(param_value mjpeg.output_width)"
  height="$(param_value mjpeg.output_height)"
  quality="$(param_value mjpeg.quality)"
  [[ "$width" == 1920 && "$height" == 1080 && "$quality" == 90 ]] && break
  sleep 1
done

echo "参数：width=${width:-<未读到>} height=${height:-<未读到>} quality=${quality:-<未读到>}"
[[ "$width" == 1920 && "$height" == 1080 && "$quality" == 90 ]] || {
  echo "高清参数未生效。远端日志：${REMOTE_LOG}" >&2
  exit 30
}

publisher_count="$(ros2 topic info /quad_tile/jpeg/compressed 2>/dev/null \
  | sed -n 's/^Publisher count: //p' | tr -d '[:space:]')"
echo "发布者数量：${publisher_count:-<未读到>}"
[[ "$publisher_count" == 1 ]] || die "相机发布者数量不是 1，拒绝继续"

echo "[3/3] 验证实际帧尺寸、帧率和 VLAHost 快照……"
VERIFY_SECONDS="$VERIFY_SECONDS" /usr/bin/python3 - <<'PY'
import io
import os
import statistics
import time

import rclpy
from PIL import Image
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

seconds = float(os.environ["VERIFY_SECONDS"])
rclpy.init()
node = Node("quadcam_quality_verifier")
samples = []
arrivals = []
qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST)

def callback(message):
    image = Image.open(io.BytesIO(bytes(message.data)))
    arrivals.append(time.monotonic())
    samples.append((image.size, len(message.data)))

node.create_subscription(CompressedImage, "/quad_tile/jpeg/compressed", callback, qos)
deadline = time.monotonic() + seconds
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.25)

try:
    print(f"frames={len(samples)}")
    if len(samples) < 2:
        raise RuntimeError("未收到足够的相机帧")
    rate = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
    sizes = sorted(set(size for size, _ in samples))
    byte_sizes = [size for _, size in samples]
    print(f"rate_hz={rate:.2f} sizes={sizes} jpeg_bytes_avg={statistics.mean(byte_sizes):.0f}")
    if sizes != [(1920, 1080)]:
        raise RuntimeError(f"实际帧尺寸不正确：{sizes}")
    if rate < 20.0:
        raise RuntimeError(f"帧率过低：{rate:.2f} Hz")
finally:
    node.destroy_node()
    rclpy.shutdown()
PY

if command -v curl >/dev/null; then
  health="$(curl --noproxy '*' -fsS --connect-timeout 3 --max-time 8 "${VLAHOST_URL}/health" 2>/dev/null || true)"
  if [[ -n "$health" ]]; then
    echo "VLAHost health：$health"
  else
    echo "警告：无法读取 VLAHost health（相机 ROS 验证仍已完成）。" >&2
  fi
fi

echo "完成：高清相机已启动并通过实际 ROS 帧验证。"
