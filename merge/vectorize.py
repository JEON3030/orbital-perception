#!/usr/bin/env python3
"""
merge.vectorize — 이진 seg 마스크 → 탐지 문서(merge.detection.v1). seg-파생 탐지.

분할(또는 유사라벨) 마스크의 연결성분을 계약의 탐지 출력으로 바꾼다. 20m 에서 탐지는
별도 탐지기가 아니라 이 벡터화가 전부다.

  · 이산 객체(선박·항공기, DET geom=obb) → 최소면적 회전상자 cv2.minAreaRect.
  · 면적 이벤트(산불·홍수·유출, geom=polygon) → 외곽선 단순화(approxPolyDP).
  · 작은 파편은 min_area_px 로 버린다(잡음·서브픽셀). 버린 개수를 결과에 남긴다.
  · 만든 문서는 돌려주기 전에 `contract.validate_detection` 으로 스스로 검증한다.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import contract

DEFAULT_MIN_AREA_PX = 4          # 이보다 작은 성분은 파편으로 버림(20m 서브픽셀 잡음)
POLY_EPS_FRAC = 0.01             # approxPolyDP 단순화 강도(둘레 대비)


def vectorize_mask(mask: np.ndarray, class_id: int, *,
                   gsd_m: float = contract.DEFAULT_GSD_M,
                   min_area_px: float = DEFAULT_MIN_AREA_PX,
                   geom: str | None = None, score: float = 1.0) -> dict:
    """이진 마스크(nonzero=객체)를 class_id 의 탐지 문서로. geom 미지정 시 클래스 기본기하."""
    if class_id not in contract.DET_BY_ID:
        raise ValueError(
            f"class_id {class_id} 는 탐지 번호표 {contract.DET_MIN_ID}~{contract.DET_MAX_ID} 밖.")
    if mask.ndim != 2:
        raise ValueError(f"마스크는 (H,W) 2차원이어야 하는데 {mask.shape} 이다.")
    geom = geom or contract.DET_BY_ID[class_id].geom
    if geom not in contract.GEOM_TYPES:
        raise ValueError(f"geom 은 {contract.GEOM_TYPES} 중 하나여야 한다(받음 {geom!r}).")
    H, W = mask.shape
    binary = (mask != 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: list[dict] = []
    dropped = 0
    for cnt in contours:
        area_px = float(cv2.contourArea(cnt))
        if area_px < min_area_px:
            dropped += 1
            continue
        d = _one(cnt, class_id, geom, area_px, gsd_m, score, H, W)
        if d is None:
            dropped += 1
            continue
        dets.append(d)

    doc = {
        "schema": contract.DETECTION_SCHEMA, "image_hw": [int(H), int(W)],
        "gsd_m": float(gsd_m), "detections": dets,
        "vectorize": {"class_id": class_id, "geom": geom,
                      "min_area_px": min_area_px, "dropped_small": dropped},
    }
    return contract.validate_detection(doc)   # 스스로 계약을 통과해야 돌려준다


def _one(cnt, class_id, geom, area_px, gsd_m, score, H, W) -> dict | None:
    base = {"class_id": class_id, "geom_type": geom, "area_px": round(area_px, 2),
            "area_m2": round(contract.detection_area_m2(area_px, gsd_m), 2),
            "score": float(score)}
    if geom == "obb":
        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
        if w <= 0 or h <= 0:
            return None                        # 퇴화(선형) 성분은 버린다
        base["obb"] = {"cx": _clip(cx, W), "cy": _clip(cy, H),
                       "w": float(w), "h": float(h), "angle_deg": float(ang)}
        return base
    # polygon: 외곽선 단순화. 3점 미만이면 버린다.
    eps = POLY_EPS_FRAC * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    base["polygon"] = [[_clip(float(x), W), _clip(float(y), H)] for x, y in approx]
    return base


def _clip(v: float, hi: int) -> float:
    """좌표를 [0,hi] 로 살짝 클램프(경계 반올림이 계약 범위를 넘지 않게)."""
    return float(min(max(v, 0.0), float(hi)))
