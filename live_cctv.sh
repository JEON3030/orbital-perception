#!/usr/bin/env bash
# 실시간 CCTV 객체탐지 런처 (run.sh/webviz.sh 와 동일 NV torch 환경).
#   ./live_cctv.sh --source "rtsp://user:pass@ip:554/stream1" [--track]
#   ./live_cctv.sh --source "http://.../cctv.m3u8"
#   → 브라우저: http://<보드IP>:7861/
export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$HOME/orbital-perception/.ultra"
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
cd "$HOME/orbital-perception"
exec "$HOME/wildfire-seg/.venv/bin/python" -u live_cctv.py "$@"
