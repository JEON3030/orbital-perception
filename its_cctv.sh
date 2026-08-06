#!/usr/bin/env bash
# 한국 ITS 공개 교통 CCTV 조회/실시간탐지 런처 (webviz.sh 와 동일 NV torch 환경).
#   ./its_cctv.sh --key <KEY> --region seoul --list
#   ./its_cctv.sh --key <KEY> --region seoul --index 0 --run
#   → 브라우저: http://<보드IP>:7861/
export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$HOME/orbital-perception/.ultra"
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
cd "$HOME/orbital-perception"
exec "$HOME/wildfire-seg/.venv/bin/python" -u its_cctv.py "$@"
