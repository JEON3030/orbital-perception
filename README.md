# Orbital Perception — 전력계측 내장 온보드 탐지/세그멘테이션

Jetson Orin Nano에서 실시간 사물/사람 **탐지(detect)** 와 **세그멘테이션(segment)** 을
수행하되, **인공위성 탑재를 목표**로 하므로 지표를 FPS가 아니라
**프레임당 에너지(mJ/frame)** 와 **frames-per-joule**로 측정한다.

## 왜 전력이 핵심인가
소형위성/큐브샛의 페이로드 전력 예산은 극도로 작다(3U ≈ 1–7W, 6U ≈ 10–30W).
Orin Nano의 **7W / 15W 모드가 정확히 이 대역**과 겹친다. 위성은 30fps가 필요 없고
(촬영 타일을 다음 촬영 전에 처리하면 됨) **에너지 효율이 곧 성능**이다.
→ 기본 운용점 **7W**, 프로그램에 tegrastats 전력 실측을 내장.

> ⚠️ Orin Nano는 rad-hard 부품이 아님 → 실제 우주용이 아니라 **지상 실증(HIL) 데모**로 프레이밍.

## 실행
```bash
./run.sh input/sample.jpg                 # 탐지 + 에너지 리포트
./run.sh input/sample.jpg --task segment  # 세그멘테이션
./run.sh input/clip.mp4 --task detect --stride 2
./run.sh 0 --task detect                  # 카메라(연결 후 /dev/video0)
```
결과: `outputs/<name>_<task>.jpg|mp4` + `outputs/<name>_<task>_energy.json`

주요 옵션: `--imgsz`(작을수록 저전력) · `--conf` · `--classes person car` ·
`--repeat`(이미지 반복추론 계측) · `--stride`(영상 프레임 스킵) · `--no-idle`.

## 전력모드 벤치마크 (7W vs 15W)
전력모드 전환엔 sudo 필요:
```bash
sudo nvpmodel -m 1 && ./bench_power.sh 7W    # 7W (ID=1)
sudo nvpmodel -m 0 && ./bench_power.sh 15W   # 15W (ID=0)
./bench_power.sh --compare                   # mJ/frame·frames/J 표
```

## 측정 예 (yolo11n, imgsz 640, 7W 모드, sample.jpg 반복추론)
| task | FPS | 평균W | mJ/frame(총) | 순수추론 mJ | frames/J |
|------|----:|-----:|-----:|-----:|-----:|
| detect  | 14.8 | 6.45 | 411 | 100 | 2.43 |
| segment | 10.8 | 6.63 | 587 | 154 | 1.71 |
(순수추론 = 유휴 4.96W 기준선을 뺀 동적 에너지)

## 파일
- `perception.py` — 메인. 탐지/세그 + 전력계측. 카메라 인덱스 입력 지원.
- `powerlog.py`   — tegrastats VDD_IN 샘플러(사다리꼴 적분 에너지). INA3221 폴백.
- `bench_power.sh`— 전력모드별 벤치 + `--compare` 표.
- `run.sh`        — NV torch 환경변수(LD_LIBRARY_PATH, PYTHONNOUSERSITE) 래퍼.

## 환경 메모
- venv는 `~/wildfire-seg/.venv`(NV torch 2.5) 재사용. ultralytics 8.4.106.
- torchvision 바이너리 없음 → `perception.py`가 NMS만 순수 torch로 스텁 주입.
- `~/.local`의 깨진 torch는 `PYTHONNOUSERSITE=1`로 무시.

## 다음 단계 (위성 방향)
- **TensorRT FP16/INT8 변환** → mJ/frame 큰 폭 감소(속도·효율 동시↑). `yolo export format=engine`.
- **위성영상 세그 훅**: `~/wildfire-seg`의 U-Net(산불 burn mask)을 `--model`로 연결,
  Sentinel-2 12밴드 타일 온보드 추론의 에너지 프로파일 측정 → 논문 그림.
- `--imgsz` 축소·모델 경량화(yolo11n→pruned)로 에너지-정확도 트레이드오프 곡선.
- 카메라(USB/CSI) 연결 시 `./run.sh 0`으로 실시간 스트림.
