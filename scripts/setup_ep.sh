#!/bin/bash
echo "============================================================"
echo " EP Package Setup — onnxruntime 변종별 격리 설치"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$SCRIPT_DIR/../ep_packages"

echo "[1/3] CPU — onnxruntime"
mkdir -p "$BASE/cpu"
pip install onnxruntime --target "$BASE/cpu" --upgrade

echo "[2/3] OpenVINO — onnxruntime-openvino"
mkdir -p "$BASE/openvino"
pip install onnxruntime-openvino --target "$BASE/openvino" --upgrade

echo "(CUDA/DirectML는 Windows 전용입니다)"

echo "============================================================"
echo " 설치 완료."
echo "============================================================"
