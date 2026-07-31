#!/usr/bin/env bash
# TensorRT 엔진 변환 래퍼. run.sh 와 동일한 NV torch 환경변수 세팅.
#   ./export_trt.sh                 # detect+segment FP16
#   ./export_trt.sh --int8 --task detect
export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$HOME/orbital-perception/.ultra"
cd "$HOME/orbital-perception"
exec "$HOME/wildfire-seg/.venv/bin/python" -u export_trt.py "$@"
