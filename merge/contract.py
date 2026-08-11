#!/usr/bin/env python3
"""
merge.contract — 두 프로그램이 만나는 자리(계약)를 코드로 못박은 곳.

박현수 「젯슨 현황과 병합 계획」의 RNR 약속 ①(클래스 번호표)·②(좌표계)와
입출력 규격을 **한 곳에서만** 정의한다. 문서가 아니라 이 파일이 원본이다 —
`python -m merge contract` 가 뽑는 표는 전부 이 상수에서 나오므로 문서-코드가
어긋날 수 없다.

계약 (형준 프로그램은 파이썬일 필요가 없다):

    python 내프로그램.py --in {in} --out {out}

    {in}   (H, W, 6) float32  .npy   위성 6밴드 반사율 [0,1]
    {out}  (H, W)    uint8     .npy   클래스 번호 맵 (0~9)

클래스 번호표는 park-hyun-su/syntax_inference `app/segdemo/targets.py` 의
LANDCOVER TargetSet 을 그대로 옮긴 것이다(id·key·한글·RGB 동일). 번호가 서로
다르면 프로그램은 안 죽고 면적 표만 그럴듯하게 틀리므로, 여기 하나로 고정한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── RNR 약속 ① : 클래스 번호표 (0~9) ──────────────────────────────────
# (id, key, 한글, "무엇인가", RGB)  — 순서가 곧 클래스 번호다.
@dataclass(frozen=True)
class LandCoverClass:
    id: int
    key: str
    ko: str
    desc: str
    rgb: tuple[int, int, int]


CLASSES: tuple[LandCoverClass, ...] = (
    LandCoverClass(0, "other",    "기타",       '"대상 아님"이 아니라 "모르는 것"',              (120, 120, 120)),
    LandCoverClass(1, "water",    "수체",       "바다·강·호수. 물 위의 배도 여기(픽셀이 담은 건 물)", (0, 100, 200)),
    LandCoverClass(2, "tree",     "산림",       "나무 수관이 덮은 면적",                          (0, 100, 0)),
    LandCoverClass(3, "grass",    "초지",       "풀·관목",                                       (255, 187, 34)),
    LandCoverClass(4, "crop",     "농지",       "논·밭",                                         (240, 150, 255)),
    LandCoverClass(5, "built_up", "시가지",     "인공피복 면적. 도로 위 차도 여기",               (250, 0, 0)),
    LandCoverClass(6, "bare",     "나지·암반",  "맨땅·바위·산지",                                (180, 180, 180)),
    LandCoverClass(7, "wetland",  "습지",       "서해안 갯벌. 수체로 합치면 갯벌이 사라진다",      (0, 150, 160)),
    LandCoverClass(8, "cloud",    "구름·그림자", "분할이 내는 값이 아님 — 구름 마스크가 따로 채움",  (100, 200, 255)),
    LandCoverClass(9, "snow",     "설빙",       "눈·얼음",                                       (240, 240, 240)),
)
N_CLASSES = len(CLASSES)                        # 10
MIN_ID = 0
MAX_ID = N_CLASSES - 1                          # 9

# 빠른 조회
BY_ID = {c.id: c for c in CLASSES}
BY_KEY = {c.key: c for c in CLASSES}
PALETTE = np.array([c.rgb for c in CLASSES], dtype=np.uint8)     # (10, 3)

# ── RNR 약속 ② : 좌표계 ───────────────────────────────────────────────
# 형준 구간(분할)은 좌표를 직접 다루지 않으므로 참고용으로만 둔다.
EXPORT_EPSG = 4326            # 내보내기(위경도)
AREA_EPSG = 32652            # 면적 계산 UTM 52N (한반도)

# ── 6밴드 입력 규격 ───────────────────────────────────────────────────
# park-hyun-su `app/segdemo/bands.py` ORDER 와 동일. NIR 은 B8A(20m)다.
BAND_ORDER: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")
BAND_S2 = {"blue": "B2", "green": "B3", "red": "B4",
           "nir": "B8A", "swir1": "B11", "swir2": "B12"}
N_BANDS = len(BAND_ORDER)                        # 6

# ── npy dtype/shape 규격 ─────────────────────────────────────────────
IN_DTYPE = np.float32
OUT_DTYPE = np.uint8
REFLECTANCE_MAX_SANE = 2.0    # 반사율은 [0,1]이 정상. 살짝 넘는 것은 허용(대기보정 잔차),
#                               크게 넘으면 0~255 원자료를 안 나눈 것이므로 경고.


class ContractError(ValueError):
    """계약 위반. 메시지에 '어디가' 틀렸는지 담는다(박현수 요구)."""


# ── 검증기 ────────────────────────────────────────────────────────────
def validate_input(arr: np.ndarray, *, path: str | None = None) -> np.ndarray:
    """입력 {in} 이 (H,W,6) float32 반사율인지 검사. 통과하면 그대로 돌려준다."""
    where = f"{path}: " if path else ""
    if arr.ndim != 3:
        raise ContractError(f"{where}입력은 (H,W,6) 3차원이어야 하는데 {arr.ndim}차원 {arr.shape} 이다.")
    if arr.shape[2] != N_BANDS:
        raise ContractError(
            f"{where}마지막 축이 밴드 {N_BANDS}개여야 하는데 {arr.shape[2]}개다. "
            f"밴드 순서는 {BAND_ORDER} (NIR=B8A). (6,H,W)로 저장했다면 축을 (H,W,6)로 옮겨라.")
    if arr.dtype != IN_DTYPE:
        raise ContractError(f"{where}dtype 이 float32 여야 하는데 {arr.dtype} 이다.")
    finite = np.isfinite(arr)
    if not finite.all():
        raise ContractError(f"{where}NaN/Inf 가 {int((~finite).sum())}개 있다. 반사율은 유한값이어야 한다.")
    hi = float(arr.max()) if arr.size else 0.0
    if hi > REFLECTANCE_MAX_SANE:
        raise ContractError(
            f"{where}최댓값이 {hi:.1f} 이다 — 반사율 [0,1] 규격을 크게 벗어난다. "
            f"0~255(또는 0~10000) 원자료를 반사율로 안 나눈 것 아닌가?")
    return arr


def validate_output(arr: np.ndarray, *, shape: tuple[int, int] | None = None,
                    path: str | None = None) -> np.ndarray:
    """출력 {out} 이 (H,W) uint8, 값이 0~9 인지 검사. shape 를 주면 크기도 맞춘다."""
    where = f"{path}: " if path else ""
    if arr.ndim != 2:
        raise ContractError(f"{where}출력은 (H,W) 2차원이어야 하는데 {arr.ndim}차원 {arr.shape} 이다.")
    if arr.dtype != OUT_DTYPE:
        # int64 로 내는 실수가 흔하다 — 값이 맞으면 어떻게 고치는지 알려준다.
        raise ContractError(
            f"{where}dtype 이 uint8 이어야 하는데 {arr.dtype} 이다. "
            f"값이 0~9 라면 `.astype('uint8')` 한 줄이면 된다.")
    if arr.size:
        lo, hi = int(arr.min()), int(arr.max())
        if lo < MIN_ID or hi > MAX_ID:
            bad = sorted(set(np.unique(arr).tolist()) - set(range(N_CLASSES)))
            raise ContractError(
                f"{where}클래스 번호가 0~9 를 벗어났다(관측 {lo}~{hi}, 규격 밖 값 {bad}). "
                f"번호표는 `python -m merge contract` 로 확인.")
    if shape is not None and tuple(arr.shape) != tuple(shape):
        raise ContractError(f"{where}출력 크기가 {tuple(arr.shape)} 인데 입력과 같은 {tuple(shape)} 여야 한다.")
    return arr


def load_input(path: str) -> np.ndarray:
    return validate_input(np.load(path), path=path)


def load_output(path: str, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    return validate_output(np.load(path), shape=shape, path=path)


def class_table_rows() -> list[tuple[int, str, str, str, str]]:
    """`contract` CLI 가 그대로 출력하는 행들. (번호, key, 한글, RGB, 설명)"""
    return [(c.id, c.key, c.ko, f"{c.rgb}", c.desc) for c in CLASSES]


# ── 탐지 출력 규격 (탐지+분할 둘 다: seg-파생 탐지) ─────────────────────────
# 20m 에서 탐지는 대형 객체만 물리적으로 가능하다(선박 200m+=10~20px, 대형기
# 40~70m=2~3px, 소형차 4m=서브픽셀 불가). 그래서 별도 탐지기가 아니라 분할 결과를
# 연결성분→벡터화한 'seg-파생 탐지'다. 출력은 인스턴스 리스트(JSON) — 선박·항공기는
# OBB(회전 상자), 재해(산불·홍수·유출)는 폴리곤. 탐지 클래스 id 는 분할 0~9 와 겹치지
# 않게 10 부터 둔다 — 번호가 섞이면 면적표가 조용히 틀리는 그 함정을 구조적으로 막는다.
DETECTION_SCHEMA = "merge.detection.v1"
DEFAULT_GSD_M = 20.0             # Sentinel-2 20m. area_m2 = area_px * gsd^2.
GEOM_TYPES: tuple[str, ...] = ("obb", "polygon")


@dataclass(frozen=True)
class DetClass:
    id: int
    key: str
    ko: str
    geom: str            # 기대 기하: "obb"(이산 객체) | "polygon"(면적 이벤트)
    desc: str


DET_CLASSES: tuple[DetClass, ...] = (
    DetClass(10, "ship",     "대형 선박",   "obb",     "항만·연안 대형 선박(200m+). 수체 대비로 잡힌다"),
    DetClass(11, "aircraft", "대형 항공기", "obb",     "공항 계류장 대형기(40~70m=2~3px, 경계). 공항 문맥 필요"),
    DetClass(12, "fire",     "산불",        "polygon", "활성 산불·연소면적(재해 이벤트, 면적 기반)"),
    DetClass(13, "flood",    "홍수",        "polygon", "침수 물 확장 면적(재해 이벤트)"),
    DetClass(14, "spill",    "유출",        "polygon", "해상 유류·오염 유출 면적(재해 이벤트)"),
)
DET_BY_ID = {c.id: c for c in DET_CLASSES}
DET_BY_KEY = {c.key: c for c in DET_CLASSES}
DET_MIN_ID = N_CLASSES                        # 10 — 분할과 겹치지 않는 첫 id
DET_MAX_ID = DET_CLASSES[-1].id               # 14


def detection_area_m2(area_px: float, gsd_m: float = DEFAULT_GSD_M) -> float:
    """픽셀 면적 → 실제 면적(m²). gsd 기본 20m."""
    return float(area_px) * float(gsd_m) * float(gsd_m)


def _finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def validate_detection(doc: dict, *, image_hw: tuple[int, int] | None = None,
                       path: str | None = None) -> dict:
    """탐지 출력({out_det}) 문서가 규격에 맞는지 검사. 통과하면 그대로 돌려준다.
    탐지 0건(빈 리스트)은 정상이다 — '선박 없음'은 오류가 아니라 결과다.
    image_hw 를 주면 분할 출력과 같은 장면인지(크기 일치) 대조한다."""
    where = f"{path}: " if path else ""
    if not isinstance(doc, dict):
        raise ContractError(f"{where}탐지 출력은 JSON 객체여야 하는데 {type(doc).__name__} 이다.")
    if doc.get("schema") != DETECTION_SCHEMA:
        raise ContractError(
            f"{where}schema 가 {doc.get('schema')!r} 이다 — {DETECTION_SCHEMA!r} 여야 한다. "
            f"탐지 출력은 이 규격의 JSON 이다(npy 아님).")
    hw = doc.get("image_hw")
    if (not isinstance(hw, (list, tuple)) or len(hw) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in hw)):
        raise ContractError(f"{where}image_hw 는 [H,W] 양의 정수여야 하는데 {hw!r} 이다.")
    H, W = int(hw[0]), int(hw[1])
    if image_hw is not None and tuple(image_hw) != (H, W):
        raise ContractError(
            f"{where}image_hw {(H, W)} 가 분할 출력 크기 {tuple(image_hw)} 와 다르다 — "
            f"같은 장면의 탐지·분할이어야 한다.")
    gsd = doc.get("gsd_m", DEFAULT_GSD_M)
    if not (_finite(gsd) and float(gsd) > 0):
        raise ContractError(f"{where}gsd_m 은 양수여야 하는데 {gsd!r} 이다.")
    dets = doc.get("detections")
    if not isinstance(dets, list):
        raise ContractError(
            f"{where}detections 는 리스트여야 하는데 {type(dets).__name__} 이다(0건이면 []).")
    for i, d in enumerate(dets):
        _validate_one_detection(d, i, H, W, where)
    return doc


def _validate_one_detection(d: dict, i: int, H: int, W: int, where: str) -> None:
    tag = f"{where}detections[{i}]: "
    if not isinstance(d, dict):
        raise ContractError(f"{tag}객체여야 하는데 {type(d).__name__} 이다.")
    cid = d.get("class_id")
    if cid not in DET_BY_ID:
        raise ContractError(
            f"{tag}class_id 가 {cid!r} 이다 — 탐지 번호표 {DET_MIN_ID}~{DET_MAX_ID} 밖이다. "
            f"(분할 0~9 와 섞지 말 것; 번호표는 `python -m merge contract`)")
    geom = d.get("geom_type")
    if geom not in GEOM_TYPES:
        raise ContractError(f"{tag}geom_type 은 {GEOM_TYPES} 중 하나여야 하는데 {geom!r} 이다.")
    if geom == "obb":
        obb = d.get("obb")
        if not isinstance(obb, dict):
            raise ContractError(f"{tag}geom_type=obb 인데 obb 딕셔너리가 없다.")
        for k in ("cx", "cy", "w", "h", "angle_deg"):
            if not _finite(obb.get(k)):
                raise ContractError(f"{tag}obb.{k} 가 유한한 수가 아니다({obb.get(k)!r}).")
        if float(obb["w"]) <= 0 or float(obb["h"]) <= 0:
            raise ContractError(f"{tag}obb 의 w,h 는 양수여야 한다(w={obb['w']}, h={obb['h']}).")
        _check_xy(float(obb["cx"]), float(obb["cy"]), H, W, tag + "obb 중심")
    else:  # polygon
        poly = d.get("polygon")
        if not isinstance(poly, list) or len(poly) < 3:
            raise ContractError(f"{tag}폴리곤은 [x,y] 점 3개 이상이어야 하는데 {poly!r} 이다.")
        for j, pt in enumerate(poly):
            if (not isinstance(pt, (list, tuple)) or len(pt) != 2
                    or not (_finite(pt[0]) and _finite(pt[1]))):
                raise ContractError(f"{tag}polygon[{j}] 가 [x,y] 유한좌표가 아니다({pt!r}).")
            _check_xy(float(pt[0]), float(pt[1]), H, W, f"{tag}polygon[{j}]")
    area = d.get("area_px")
    if not (_finite(area) and float(area) > 0):
        raise ContractError(f"{tag}area_px 는 양수여야 하는데 {area!r} 이다.")
    sc = d.get("score")
    if sc is not None and not (_finite(sc) and 0.0 <= float(sc) <= 1.0):
        raise ContractError(f"{tag}score 는 0~1 이어야 하는데 {sc!r} 이다.")


def _check_xy(x: float, y: float, H: int, W: int, what: str) -> None:
    if not (0 <= x <= W and 0 <= y <= H):
        raise ContractError(
            f"{what} ({x},{y}) 가 이미지 {W}x{H} 밖이다 — 픽셀좌표가 맞나"
            f"(위경도·정규화 0~1 아닌지)?")


def load_detection(path: str, *, image_hw: tuple[int, int] | None = None) -> dict:
    return validate_detection(json.loads(Path(path).read_text()), image_hw=image_hw, path=path)


def det_class_table_rows() -> list[tuple[int, str, str, str, str]]:
    """`contract` CLI 가 출력하는 탐지 클래스 행. (번호, key, 한글, 기하, 설명)"""
    return [(c.id, c.key, c.ko, c.geom, c.desc) for c in DET_CLASSES]
