#!/usr/bin/env bash
# Orbital Perception 시각화 웹앱 런처. run.sh 와 동일한 NV torch 환경을 세팅한다.
#   ./webviz.sh                # http://<이 보드 IP>:7860
#   ./webviz.sh --port 8000
export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1          # ~/.local의 깨진 torch/torchvision 무시
export YOLO_CONFIG_DIR="$HOME/orbital-perception/.ultra"   # ultralytics 설정 격리
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1                          # SegFormer(HF) 텔레메트리 끔
cd "$HOME/orbital-perception"
exec "$HOME/wildfire-seg/.venv/bin/python" -u webviz.py "$@"
