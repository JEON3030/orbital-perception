#!/usr/bin/env bash
# SegFormer → TensorRT FP16 엔진 변환 런처 (run.sh/webviz.sh 와 동일 NV torch 환경).
#   ./export_seg_trt.sh                 # road, aerial 둘 다
#   ./export_seg_trt.sh --domain aerial
export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$HOME/orbital-perception/.ultra"
export HF_HUB_DISABLE_TELEMETRY=1
cd "$HOME/orbital-perception"
exec "$HOME/wildfire-seg/.venv/bin/python" -u export_seg_trt.py "$@"
