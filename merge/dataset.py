#!/usr/bin/env python3
"""
merge.dataset — 재해 유사라벨 학습 데이터셋 빌더. 이벤트 → 취득 → 유사라벨 → 타일.

위성 전용 전환 ③의 학습데이터층. 기본 재해 이벤트(산불 pre+post dNBR, 홍수 MNDWI)를
`acquire` 로 독립 취득해 `labels` 로 유사라벨을 만들고, 학습용 (tile,tile,6)+label 타일로
자른다. 타일은 npz(x: 6밴드 float32, y: uint8), manifest.json 에 출처·유사라벨 방식·통계를
남긴다(정직성 — 유사라벨은 진짜 GT 아님을 기록).

Steven/ 자료·코드는 쓰지 않는다. 이벤트는 공개정보(위치·시기)일 뿐이고 장면은 acquire 로
새로 받는다. 이벤트별 실패(장면 0건 등)는 fail-loud 로 기록하되 나머지는 계속 만든다.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from . import acquire, labels

# 기본 이벤트: 산불(dNBR pre+post)·홍수(MNDWI). bbox=(minlon,minlat,maxlon,maxlat).
# 장면은 acquire 가 조건 내 가장 맑은 것을 고른다. 시기는 pre=식생기(leaf-on) 관례.
DEFAULT_EVENTS = [
    {"name": "uljin_2022_burn", "kind": "burn",
     "bbox": [129.22, 36.92, 129.44, 37.14],
     "pre": ["2021-04-10", "2021-05-25"], "post": ["2022-03-20", "2022-05-10"], "cloud": 30},
    {"name": "nakdong_delta_water", "kind": "water",
     "bbox": [128.85, 35.02, 129.05, 35.22],
     "post": ["2024-01-01", "2024-04-30"], "cloud": 15},
]
DET_CLASS = {"burn": 12, "water": 13}      # 유사라벨 종류 → 계약 탐지 클래스(fire/flood)


def tile_arrays(cube: np.ndarray, label: np.ndarray, *, tile: int = 256,
                stride: int | None = None, keep_empty: bool = True,
                min_pos: float = 0.0) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """(H,W,6) 큐브·(H,W) 라벨을 tile×tile 로 자른다. 가장자리 잔여(부분 타일)는 버린다.
    keep_empty=False 면 양성비율 min_pos 미만 타일을 뺀다. (x, y, pos_frac) 리스트."""
    if cube.shape[:2] != label.shape:
        raise ValueError(f"큐브 {cube.shape[:2]} 와 라벨 {label.shape} 크기가 다르다.")
    stride = stride or tile
    H, W = label.shape
    out = []
    for y0 in range(0, H - tile + 1, stride):
        for x0 in range(0, W - tile + 1, stride):
            y = label[y0:y0 + tile, x0:x0 + tile]
            pos = float(y.mean())
            if not keep_empty and pos < min_pos:
                continue
            out.append((cube[y0:y0 + tile, x0:x0 + tile, :], y, pos))
    return out


def build_event(ev: dict, out_dir: str, *, tile: int = 256,
                stride: int | None = None) -> dict:
    """이벤트 하나를 취득→유사라벨→타일 저장. 통계 dict 반환."""
    kind = ev["kind"]
    if kind not in DET_CLASS:
        raise ValueError(f"이벤트 kind 는 {list(DET_CLASS)} 중 하나여야 한다(받음 {kind!r}).")
    bbox = tuple(ev["bbox"])
    pre_id = None
    with tempfile.TemporaryDirectory() as td:
        post_meta = acquire.acquire_scene(bbox, ev["post"][0], ev["post"][1],
                                          cloud_max=ev.get("cloud", 20),
                                          out_npy=str(Path(td) / "post.npy"))
        post = np.load(str(Path(td) / "post.npy"))
        if kind == "burn":
            pre_meta = acquire.acquire_scene(bbox, ev["pre"][0], ev["pre"][1],
                                             cloud_max=ev.get("cloud", 20),
                                             out_npy=str(Path(td) / "pre.npy"))
            pre = np.load(str(Path(td) / "pre.npy"))
            pre, post = _align(pre, post)
            label = labels.pseudo_burn(post, pre=pre)
            pre_id = pre_meta["scene_id"]
            pseudo = "dNBR>=0.27 (USGS, pre+post)"
        else:
            label = labels.pseudo_water(post)
            pseudo = "MNDWI>0 (Xu)"

    tiles = tile_arrays(post, label, tile=tile, stride=stride)
    ev_dir = Path(out_dir) / ev["name"]
    ev_dir.mkdir(parents=True, exist_ok=True)
    for i, (x, y, _pos) in enumerate(tiles):
        np.savez_compressed(ev_dir / f"tile_{i:04d}.npz", x=x, y=y)
    return {
        "event": ev["name"], "kind": kind, "det_class": DET_CLASS[kind],
        "pseudo_label": pseudo, "tiles": len(tiles),
        "pos_frac": round(float(label.mean()), 4), "scene_shape": list(post.shape),
        "scene_post": post_meta["scene_id"], "scene_pre": pre_id,
        "cloud_post": post_meta.get("cloud_cover"),
    }


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pre/post 를 공통 최소격자로 자른다(워프 1px 오차 방어)."""
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def build_dataset(events: list[dict], out_dir: str, *, tile: int = 256,
                  stride: int | None = None) -> dict:
    """이벤트 리스트로 데이터셋 구축. 이벤트 실패는 기록하고 계속. manifest 반환·저장."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = []
    for ev in events:
        try:
            per.append(build_event(ev, out_dir, tile=tile, stride=stride))
        except (acquire.AcquireError, ValueError) as e:
            per.append({"event": ev.get("name", "?"), "error": str(e), "tiles": 0})
    manifest = {
        "tile": tile, "total_tiles": sum(p.get("tiles", 0) for p in per),
        "classes": {"12": "fire (dNBR pseudo-label)", "13": "flood (MNDWI pseudo-label)"},
        "note": "유사라벨(pseudo-label) 기반 — 진짜 GT 아님. 지수·임계는 물리 근사.",
        "events": per,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
