#!/usr/bin/env bash
# Orbital Perception 실행 래퍼. NV torch용 환경변수 세팅 후 perception.py 호출.
#   ./run.sh input/photo.jpg
#   ./run.sh input/clip.mp4 --task segment
export LD_LIBRARY_PATH="$HOME/wildfire-seg/libs:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1          # ~/.local의 깨진 torch/torchvision 무시
export YOLO_CONFIG_DIR="$HOME/orbital-perception/.ultra"   # ultralytics 설정 격리
cd "$HOME/orbital-perception"
exec "$HOME/wildfire-seg/.venv/bin/python" -u perception.py "$@"
