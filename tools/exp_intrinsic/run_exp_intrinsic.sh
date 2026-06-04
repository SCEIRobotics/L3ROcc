#!/usr/bin/env bash
# 运行 Pi3X RGB-only vs RGB+intrinsic 的 最小修改对比实验
#
# 用法:
#   bash tools/exp_intrinsic/run_exp_intrinsic.sh                          # 使用默认路径(GPU)
#   bash tools/exp_intrinsic/run_exp_intrinsic.sh <video> <intr> <out_dir> # 自定义路径
#   USE_CPU=1 bash tools/exp_intrinsic/run_exp_intrinsic.sh                # 强制 CPU
#   MAX_FRAMES=8 INTERVAL=2 bash tools/exp_intrinsic/run_exp_intrinsic.sh   # 调整帧数 / 抽样间隔（默认10抽1）

set -euo pipefail

# 项目根目录(脚本所在目录向上两级)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- 默认路径(按需修改) ----------
DEFAULT_VIDEO="/mnt/data_ssd/share/datasets/vln_data/vln_real_data/lerobot_data/20260601/rosbag_20260529_155555/videos/chunk-000/observation.images.RGB/episode_000.mp4"
DEFAULT_INTR="/mnt/data_ssd/share/datasets/vln_data/vln_real_data/lerobot_data/20260601/rosbag_20260529_155555/meta/info.json"
DEFAULT_OUT="${SCRIPT_DIR}/exp_out/episode_000"

VIDEO_PATH="${1:-$DEFAULT_VIDEO}"
INTR_PATH="${2:-$DEFAULT_INTR}"
OUT_DIR="${3:-$DEFAULT_OUT}"

# ---------- 可选环境变量 ----------
MAX_FRAMES="${MAX_FRAMES:-12}"
INTERVAL="${INTERVAL:-10}"
USE_CPU="${USE_CPU:-0}"

EXTRA_ARGS=()
if [[ "${USE_CPU}" == "1" ]]; then
    EXTRA_ARGS+=("--cpu")
    echo "[mode] CPU (forced via USE_CPU=1)"
else
    echo "[mode] GPU (set USE_CPU=1 to force CPU)"
fi

echo "[paths]"
echo "  video_path = ${VIDEO_PATH}"
echo "  intr_path  = ${INTR_PATH}"
echo "  out_dir    = ${OUT_DIR}"
echo "  max_frames = ${MAX_FRAMES}"
echo "  interval   = ${INTERVAL}"
echo

# ---------- 前置检查 ----------
if [[ ! -f "${VIDEO_PATH}" ]]; then
    echo "[error] video not found: ${VIDEO_PATH}" >&2
    exit 1
fi
if [[ ! -f "${INTR_PATH}" ]]; then
    echo "[error] intrinsic json not found: ${INTR_PATH}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

cd "${PROJECT_ROOT}"
python tools/exp_intrinsic/exp_intrinsic_compare.py \
    --video_path "${VIDEO_PATH}" \
    --intr_path  "${INTR_PATH}" \
    --out_dir    "${OUT_DIR}" \
    --max_frames "${MAX_FRAMES}" \
    --interval   "${INTERVAL}" \
    "${EXTRA_ARGS[@]}"

echo
echo "[done] outputs:"
echo "  ${OUT_DIR}/report.json"
echo "  ${OUT_DIR}/A_rgb_only_frame0.ply"
echo "  ${OUT_DIR}/B_rgb_plus_intr_frame0.ply"
