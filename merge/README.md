# merge — 계약 · 계측 · 채택 (구간02 ↔ 구간03 병합 도구)

형준(구간02, 온보드 분할)과 박현수(구간03, 전력) 두 프로그램을 **같은 자로 재고
항목별로 채택**하기 위한 도구다. 박현수 「젯슨 현황과 병합 계획」에서 이름만 있고
아직 구현되지 않았던 `segdemo.jetson`(계측)·`segdemo.merge`(채택)를,
`orbital-perception` 인프라(`powerlog`·`device`)를 재사용해 **정정본**으로 만들었다.

**모델 무관.** 형준 프로그램은 파이썬이 아니어도 된다 — `--seg-cmd` 로 실행 명령을
그대로 끼운다. 붙는 자리를 파일(npy in/out)로 뒀기 때문이다.

## 왜 이 도구인가 — 담긴 보완점(정정 포함)

| 보완점 | 어디에 |
|---|---|
| **정확도 실측(저울의 반쪽)** — 전력 축은 idle 차감·스로틀·표A 게이트로 엄밀히 재면서 정작 형준의 기여 전부인 **분할 정확도**는 `--miou` 로 손입력받아 왔다. 이제 예측·정답 npy 로 per-class IoU·mIoU 를 **실측**하고, 그 값을 `adopt` 가 손입력 대신 읽는다. 지표 수식은 `metrics.ConfusionMatrix` 를 재사용(계약처럼 지표도 한 곳에서만). | `score.py`, `adopt --score-a/-b` |
| **에너지 회계 정정** — total 과 dynamic(idle 차감) mJ/frame 을 **항상 같이** 낸다. "2배 빨라도 전기 7%"는 board-on-window 레짐에서만 참이고, 촬영→처리→대기 듀티사이클에선 속도가 곧 에너지다. 두 수치가 나란히 있어야 그 판단을 한다. | `meter.py` (`powerlog.idle_watt` 차감) |
| **GPU fail-loud** — GPU 를 요구했는데 없으면 조용히 CPU 로 안 내려가고 멈춘다. | `meter.require_gpu_or_die` |
| **10-클래스 계약을 한 곳에서만** — 번호가 다르면 프로그램은 안 죽고 면적 표만 틀린다. | `contract.CLASSES` (park-hyun-su `targets.LANDCOVER` 이식) |
| **npy 계약 검증** — 어디가 틀렸는지 말해준다(int64→uint8, 밴드 수, 범위…). | `contract.validate_input/output` |
| **표 A(고정 조건) 캡처 + 게이트** — 보드·전력모드·빈 메모리·반복을 결과에 남기고, 빈 메모리 부족 시 "성능이 아니라 운"이라 측정 거부. | `meter.capture_table_a` |
| **열 스로틀 경고** — 회차가 느려지면(36→64→81 함정) 플래그. | `meter._throttle_flag` |
| **표 B 채택** — 5% 동률→기존 유지, 미측정→**"판정 불가"(0 아님)**, 조건 불일치→중단, 항목별 채택, 진 쪽은 baseline 으로. | `adopt.py` |

## 빠른 사용

```bash
cd ~/orbital-perception

# 0) 계약을 코드에서 출력 (문서-코드가 어긋날 수 없다)
python3 -m merge contract

# 0.5) 위성 장면 취득 — S2 20m 6밴드 npy를 계약대로 (STAC+GDAL, 인증 불필요)
python3 -m merge acquire --bbox 129.02,35.06,129.10,35.12 \
  --start 2024-02-01 --end 2024-04-30 --cloud 15 --out scene.npy
#   AOI 창만 range-read + 20m 리샘플. 가장 맑은 장면 자동선택. +scene.provenance.json

# 1) 낸 결과(또는 취득 npy)가 규격에 맞는지 스스로 확인
python3 -m merge check scene.npy --kind in            # 입력(6밴드) 검사
python3 -m merge check 결과.npy --shape 2048,2048     # 출력(분할) 검사
python3 -m merge check det.json --kind det            # 탐지({out_det}) 검사

# 1.5) 재해 유사라벨 → seg-파생 탐지 (손라벨 없이, 모델 자리엔 유사라벨/모델출력 무엇이든)
python3 -m merge label scene.npy --target water --out water.npy          # 홍수: MNDWI>0
python3 -m merge label post.npy --target burn --pre pre.npy --out burn.npy  # 산불: dNBR≥0.27
python3 -m merge detect water.npy --class-id 13 --min-area 25 --out flood.json  # 물→홍수 폴리곤
python3 -m merge check flood.json --kind det          # seg-파생 탐지가 계약에 맞는지

# 2) 유휴 전력 → 이 값이 있어야 dynamic(순수) mJ/frame 이 나온다
python3 -m merge idle

# 3) 같은 자로 재기 — 형준 명령만 바꿔 끼운다
python3 -m merge measure --name 우리 \
  --seg-cmd "python seg_ours.py --in {in} --out {out}" \
  --in scene2048.npy --repeat 3 --measure-idle --out 우리.json

python3 -m merge measure --name 형준 \
  --seg-cmd "python 형준프로그램.py --in {in} --out {out}" \
  --in scene2048.npy --repeat 3 --measure-idle --out 형준.json
#   GPU 를 요구하려면 --require-gpu (CUDA 없으면 멈춘다)
#   메모리 게이트 기본 2000MB, 조정은 --min-free-mb

# 4) 정확도(mIoU)를 실측한다 — 손입력 대신 예측·정답 npy 로 잰다
python3 -m merge score 우리_pred.npy label.npy --name 우리 --out score_a.json
python3 -m merge score 형준_pred.npy label.npy --name 형준 --out score_b.json
#   {pred} (H,W) uint8 계약 준수, {label} (H,W) 정수(무효화소 255/-1 허용)

# 5) 항목별 채택표 — 정확도는 --score-a/-b 로 실측 연동(손입력 아님)
python3 -m merge adopt 우리.json 형준.json \
  --name-a 우리 --name-b 형준 --incumbent 우리 \
  --score-a score_a.json --score-b score_b.json
#   라벨이 없어 손으로 넣어야 하면 --miou-a/-b 로 덮어쓸 수 있다(명시 우선)
```

## 계약 (형준이 맞출 것은 이 규격이 전부)

```
python 내프로그램.py --in {in} --out {out}
  {in}   (H, W, 6) float32 .npy   밴드순서 [blue,green,red,nir=B8A,swir1,swir2], 반사율 [0,1]
  {out}  (H, W)    uint8   .npy   분할 클래스 0~9 (표는 `merge contract`)
```

**탐지+분할 둘 다** 하면 분할 출력에 더해 탐지 출력을 낸다(seg-파생 탐지):

```
  {out_det}  merge.detection.v1 JSON   탐지 인스턴스 리스트
    {"schema":"merge.detection.v1", "image_hw":[H,W], "gsd_m":20.0,
     "detections":[
       {"class_id":10, "geom_type":"obb",
        "obb":{"cx":.,"cy":.,"w":.,"h":.,"angle_deg":.}, "area_px":., "score":.},   # 선박·항공기
       {"class_id":13, "geom_type":"polygon",
        "polygon":[[x,y],…], "area_px":., "score":.}                                # 재해(홍수 등)
     ]}
```
- 탐지 클래스 10~14(선박·항공기·산불·홍수·유출) — **분할 0~9 와 겹치지 않는다**(면적표 오염 방지).
- 좌표는 픽셀(0≤x≤W, 0≤y≤H), 면적 `area_m² = area_px × gsd²`. **탐지 0건(`[]`)은 정상**.
- ⚠️ 20m 물리한계: 탐지는 대형객체만(선박 200m+ 가능, 대형기 2~3px 경계, 소형차 불가). `merge contract` 참고.

## 검증

```bash
python3 -m merge selftest          # 젯슨 없이 계약(분할·탐지)·채택·정확도 수식 자체검증
pytest merge/tests -q              # 순수 로직 + e2e(dummy_seg 로 measure 배관 실검증)
```
실측 예(이 Orin Nano, 7W, idle 5.13W): 같은 입력에 느린 쪽이 total 2193→3219 mJ/frame
이지만 그 차이는 idle 지배분이고, **순수(dynamic) 236→285 mJ/frame** 이 실제 계산 비용이다.
→ 총 에너지만 보면 "속도 무의미"로 오해하고, dynamic 을 같이 봐야 "속도=에너지"가 보인다.

## 재사용/경계

- `powerlog.PowerMeter`, `device`(fail-loud 장치 선택), `metrics.ConfusionMatrix`(mIoU) 는 `orbital-perception` 것을 그대로 쓴다 — 지표를 복제하면 두 값이 갈린다.
- 계약 상수(클래스표·밴드)는 park-hyun-su `app/segdemo/targets.py`·`bands.py` 를 이식했다(출처 주석).
- **Steven/ 자료는 쓰지 않는다**(데이터 소스 포함) — 지시에 따라 제외.
- 이 계측기는 **보드 총전력만 진실로 잰다.** 외부 `--seg-cmd` 프로그램이 실제로 GPU 를
  썼는지는 그 프로그램이 스스로 밝혀야 한다(fail-loud 는 그쪽 몫). 이 경계를 결과에 명시한다.
```
