#!/usr/bin/env bash
# 시맨틱 세그(SegFormer) 실행 래퍼. run.sh 와 같은 NV torch 환경 + _deps 격리 스택.
#   ./semantic.sh input/sample.jpg
#   ./semantic.sh input/sat.png --domain aerial --device cuda
export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$HOME/orbital-perception/.ultra"
export HF_HUB_DISABLE_TELEMETRY=1
cd "$HOME/orbital-perception"
exec "$HOME/wildfire-seg/.venv/bin/python" -u semantic.py "$@"
