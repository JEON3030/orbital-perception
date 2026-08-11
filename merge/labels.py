#!/usr/bin/env python3
"""
merge.labels — 분광지수 유사라벨(pseudo-label). 재해(홍수·산불) 학습 정답을 손라벨 없이.

위성사진 전용 전환 ③의 토대. 6밴드 계약 입력에서 물리 기반 분광지수를 계산해, 문헌
임계로 재해 마스크를 만든다. 이게 세그 모델의 **학습 정답**이 되고, 뒤이어 이 마스크를
`vectorize` 로 탐지 문서로 바꾼다(seg-파생 탐지).

  · 물(홍수 확장) = MNDWI>0 (Xu; SWIR 로 시가지 오탐을 줄임). NDWI(McFeeters)도 선택.
  · 불(연소면적):
      - 이시기(pre 있음): dNBR = NBR_pre − NBR_post ≥ 0.27 (USGS moderate-high). 정공법.
      - 단시기(pre 없음): NBR<0 & NDVI<0.2 근사 — 맨땅과 헷갈려 **약하다**(경고).

⚠️ 유사라벨은 진짜 GT 가 아니다. 지수·임계가 물리적으로 그럴듯할 뿐, 구름·그림자·얕은물·
탁도에 흔들린다. 보고 시 '유사라벨 기반'임을 명시하고, 가능하면 실측으로 검증한다.
"""
from __future__ import annotations

import numpy as np

from . import contract

# 계약 밴드 위치: 0 blue, 1 green, 2 red, 3 nir(B8A), 4 swir1(B11), 5 swir2(B12)
_B = {b: i for i, b in enumerate(contract.BAND_ORDER)}

USGS_DNBR_MODERATE = 0.27       # USGS dNBR moderate-high 연소
DEFAULT_NBR_BURN = 0.0          # 단시기 근사: NBR 이보다 낮으면 연소 의심
DEFAULT_NDVI_BARE = 0.2         # 단시기 근사: 식생 거의 없음


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """정규화 차분지수 (a−b)/(a+b). 분모 0 은 0 으로(0으로 나눔 방지)."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    denom = a + b
    out = np.zeros_like(denom, dtype=np.float32)
    np.divide(a - b, denom, out=out, where=denom != 0)
    return out


def spectral_index(cube: np.ndarray, name: str) -> np.ndarray:
    """계약 6밴드 큐브에서 정규화지수를 낸다. name ∈ {ndwi, mndwi, ndvi, nbr}."""
    cube = contract.validate_input(cube)
    g, r, nir, sw1, sw2 = (cube[..., _B["green"]], cube[..., _B["red"]], cube[..., _B["nir"]],
                           cube[..., _B["swir1"]], cube[..., _B["swir2"]])
    if name == "ndwi":                       # McFeeters 물
        return _ratio(g, nir)
    if name == "mndwi":                      # Xu 물(SWIR)
        return _ratio(g, sw1)
    if name == "ndvi":                       # 식생
        return _ratio(nir, r)
    if name == "nbr":                        # 연소비
        return _ratio(nir, sw2)
    raise ValueError(f"모르는 지수 {name!r} — {{ndwi, mndwi, ndvi, nbr}} 중 하나.")


def pseudo_water(cube: np.ndarray, *, index: str = "mndwi", thr: float = 0.0) -> np.ndarray:
    """물(홍수 확장) 유사라벨. index>thr 인 화소를 1(water)로. (H,W) uint8."""
    if index not in ("mndwi", "ndwi"):
        raise ValueError(f"물 지수는 mndwi/ndwi 중 하나여야 한다(받음 {index!r}).")
    return (spectral_index(cube, index) > thr).astype(np.uint8)


def pseudo_burn(cube: np.ndarray, *, pre: np.ndarray | None = None,
                dnbr_thr: float = USGS_DNBR_MODERATE,
                nbr_thr: float = DEFAULT_NBR_BURN,
                ndvi_thr: float = DEFAULT_NDVI_BARE) -> np.ndarray:
    """연소면적 유사라벨. pre(이전 장면)가 있으면 dNBR≥thr(정공법), 없으면 단시기 근사.
    (H,W) uint8, 1=burn."""
    nbr_post = spectral_index(cube, "nbr")
    if pre is not None:
        pre = contract.validate_input(pre)
        if pre.shape != cube.shape:
            raise ValueError(f"pre {pre.shape} 와 post {cube.shape} 크기가 다르다 — 같은 격자여야.")
        dnbr = spectral_index(pre, "nbr") - nbr_post
        return (dnbr >= dnbr_thr).astype(np.uint8)
    # 단시기 근사(약함): 낮은 NBR + 낮은 NDVI
    ndvi = spectral_index(cube, "ndvi")
    return ((nbr_post < nbr_thr) & (ndvi < ndvi_thr)).astype(np.uint8)


def coverage(mask: np.ndarray) -> float:
    """마스크에서 1의 비율(0~1). 커버리지 점검·정직성 보고용."""
    return float(mask.mean()) if mask.size else 0.0
