#!/usr/bin/env python3
"""
merge.acquire — Sentinel-2 20m 취득: STAC 검색 → COG 창읽기 → 계약 6밴드 npy.

위성사진 전용 전환의 취득층. 형준 `--seg-cmd` 가 먹을 `{in}` (H,W,6) float32 반사율
npy 를 계약(`contract.BAND_ORDER`, 20m)대로 만든다.

  · STAC(earth-search, 인증 불필요)로 AOI·기간·구름 조건에 맞는 S2 L2A 장면을 찾고,
  · GDAL /vsicurl 로 **필요한 AOI 창만 range-read** + 20m 리샘플(`gdalwarp -r average`),
  · STAC raster 메타의 scale/offset 으로 DN→반사율(처리 베이스라인 04.00 오프셋도 반영),
  · 계약 밴드순으로 스택 → `contract.validate_input` 로 **스스로 규격을 확인**한 뒤 저장.

순수 핵심(스택·반사율·STAC 파싱)은 네트워크 없이 단위검증 가능하고, 검색·다운로드는
얇은 fail-loud 계층이다 — `meter` 가 `run_seg_cmd` 를 분리한 것과 같은 구조. 조용한
실패(빈 결과·다운로드 오류)를 넘기지 않고 멈춘다.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import contract

# earth-search v1 sentinel-2-l2a 자산키 → 계약 밴드키. NIR 은 B8A(nir08, 20m)다.
STAC_URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
ASSET_KEYS = {
    "blue": "blue",       # B02 10m
    "green": "green",     # B03 10m
    "red": "red",         # B04 10m
    "nir": "nir08",       # B8A 20m  (B08 "nir" 10m 아님 — 계약이 B8A)
    "swir1": "swir16",    # B11 20m
    "swir2": "swir22",    # B12 20m
}
DEFAULT_SCALE = 0.0001    # DN→반사율 기본(=1/10000). STAC 메타가 있으면 그걸 쓴다.
DEFAULT_OFFSET = 0.0
TARGET_GSD_M = 20.0

# ⚠️ 반사율 오프셋 — 경험적 결정(실측 확인 2026-08-11).
# earth-search 'sentinel-cogs' COG 의 화소 DN 은 baseline 04.00 의 +1000 시프트가 이미
# 반영돼 있지 **않다**(소양호 장면 green DN 중앙값 346 < 1000). 그런데 STAC raster:bands
# 메타는 일괄로 offset -0.1(=-1000 DN)을 준다. 이걸 그대로 빼면 실제 신호(특히 물의 낮은
# green/swir)가 음수→0 으로 뭉개져 MNDWI 물탐지가 6.7%→0.1% 로 무너진다. 그래서 이 소스에서는
# STAC offset 을 적용하지 않고 scale 만 쓴다(offset 0). DN 은 이미 음수가 없어 clip 도 무해.
APPLY_STAC_OFFSET = False


class AcquireError(RuntimeError):
    """취득 실패(검색 0건·다운로드 오류·밴드 누락 등)."""


# ── 순수 핵심 (네트워크 없이 단위검증 가능) ──────────────────────────────────
def dn_to_reflectance(dn: np.ndarray, *, scale: float = DEFAULT_SCALE,
                      offset: float = DEFAULT_OFFSET, clip: bool = True) -> np.ndarray:
    """정수 DN → float32 반사율. refl = DN*scale + offset. 음수(무효/오프셋 잔차)는 0으로."""
    r = dn.astype(np.float32) * np.float32(scale) + np.float32(offset)
    if clip:
        np.clip(r, 0.0, None, out=r)         # 반사율 하한 0 (baseline 04.00 오프셋 잔차 방지)
    return r


def stack_to_contract(bands: dict[str, np.ndarray]) -> np.ndarray:
    """밴드키→2D 반사율 dict 를 계약 밴드순 (H,W,6) float32 로 쌓고 규격을 검증한다."""
    missing = [b for b in contract.BAND_ORDER if b not in bands]
    if missing:
        raise AcquireError(f"밴드가 빠졌다: {missing}. 계약 순서는 {contract.BAND_ORDER}.")
    ref = bands[contract.BAND_ORDER[0]]
    for b in contract.BAND_ORDER:
        a = bands[b]
        if a.ndim != 2:
            raise AcquireError(f"밴드 {b} 는 (H,W) 2차원이어야 하는데 {a.shape} 이다.")
        if a.shape != ref.shape:
            raise AcquireError(
                f"밴드 크기가 다르다: {b} {a.shape} vs {contract.BAND_ORDER[0]} {ref.shape} "
                f"— 같은 20m 격자로 워프됐는지 확인.")
    cube = np.stack([bands[b].astype(np.float32) for b in contract.BAND_ORDER], axis=-1)
    return contract.validate_input(cube)     # 계약을 스스로 통과해야 저장한다


def pick_least_cloudy(items: list[dict]) -> dict:
    """STAC feature 리스트에서 구름이 가장 적은 장면을 고른다. 비면 fail-loud."""
    if not items:
        raise AcquireError("조건에 맞는 S2 장면이 0건 — 기간·AOI·구름 상한을 넓혀라.")
    def cloud(it):
        return it.get("properties", {}).get("eo:cloud_cover", 100.0)
    return sorted(items, key=cloud)[0]


def band_hrefs(item: dict) -> dict[str, dict]:
    """계약 밴드키 → {href, scale, offset}. 자산·scale/offset 을 STAC 메타에서 뽑는다."""
    assets = item.get("assets", {})
    out: dict[str, dict] = {}
    for bkey, akey in ASSET_KEYS.items():
        a = assets.get(akey)
        if not a or "href" not in a:
            raise AcquireError(f"자산 {akey}({bkey}) 가 STAC 아이템에 없다 — 컬렉션이 맞나?")
        rb = (a.get("raster:bands") or [{}])[0]
        out[bkey] = {
            "href": a["href"],
            "scale": float(rb.get("scale", DEFAULT_SCALE)),
            # STAC offset 은 이 소스에서 신호를 뭉갠다(위 APPLY_STAC_OFFSET 주석) → 0 사용.
            "offset": float(rb.get("offset", DEFAULT_OFFSET)) if APPLY_STAC_OFFSET else 0.0,
        }
    return out


# ── 네트워크 계층 (fail-loud) ────────────────────────────────────────────
def search(bbox: tuple[float, float, float, float], start: str, end: str, *,
           cloud_max: float = 20.0, nodata_max: float = 10.0,
           collection: str = COLLECTION, limit: int = 50) -> list[dict]:
    """earth-search STAC 검색. bbox=(minlon,minlat,maxlon,maxlat), 날짜 ISO(YYYY-MM-DD).
    nodata_max: 타일 nodata 비율 상한 — 궤도 가장자리의 '반쪽 장면'(대부분 nodata)을 거른다.
    이게 없으면 AOI 대부분이 0으로 채워져 유사라벨이 nodata 에 희석된다."""
    import requests
    body = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": cloud_max},
                  "s2:nodata_pixel_percentage": {"lte": nodata_max}},
        "limit": limit,
    }
    try:
        r = requests.post(STAC_URL, json=body, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise AcquireError(f"STAC 검색 실패: {e}")
    return r.json().get("features", [])


def _vsi_path(href: str) -> str:
    if href.startswith("s3://"):
        return "/vsis3/" + href[len("s3://"):]
    return "/vsicurl/" + href


def warp_band_to_20m(href: str, bbox: tuple[float, float, float, float], utm_epsg: int,
                     out_tif: str) -> np.ndarray:
    """COG 에서 AOI 창만 20m UTM 격자로 워프(range-read)해 DN 배열을 돌려준다."""
    minlon, minlat, maxlon, maxlat = bbox
    cmd = [
        "gdalwarp", "-q", "-overwrite",
        "-t_srs", f"EPSG:{utm_epsg}",
        "-te", str(minlon), str(minlat), str(maxlon), str(maxlat), "-te_srs", "EPSG:4326",
        "-tr", str(TARGET_GSD_M), str(TARGET_GSD_M), "-r", "average",
        _vsi_path(href), out_tif,
    ]
    env = dict(os.environ,
               AWS_NO_SIGN_REQUEST="YES", AWS_REGION="us-west-2",
               GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        raise AcquireError(f"gdalwarp 300s 초과: {href}")
    if p.returncode != 0:
        raise AcquireError(f"gdalwarp 실패({p.returncode}): {href}\n  {p.stderr.strip()[:300]}")
    import rasterio
    with rasterio.open(out_tif) as ds:
        return ds.read(1)


def valid_fraction(cube: np.ndarray) -> float:
    """유효(비-nodata) 화소 비율. nodata 는 전 밴드가 0 인 화소로 본다."""
    return float((cube != 0).any(axis=-1).mean()) if cube.size else 0.0


def acquire_scene(bbox: tuple[float, float, float, float], start: str, end: str, *,
                  cloud_max: float = 20.0, nodata_max: float = 10.0,
                  collection: str = COLLECTION, out_npy: str | None = None) -> dict:
    """AOI·기간으로 가장 맑은 S2 장면을 받아 계약 6밴드 npy 를 만든다. 프로버넌스 dict 반환."""
    items = search(bbox, start, end, cloud_max=cloud_max, nodata_max=nodata_max,
                   collection=collection)
    item = pick_least_cloudy(items)
    props = item.get("properties", {})
    utm = props.get("proj:epsg")
    if not utm:
        raise AcquireError("STAC 아이템에 proj:epsg 가 없다 — 20m UTM 격자를 정할 수 없다.")
    hrefs = band_hrefs(item)

    bands: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as td:
        for bkey in contract.BAND_ORDER:
            info = hrefs[bkey]
            dn = warp_band_to_20m(info["href"], bbox, int(utm), str(Path(td) / f"{bkey}.tif"))
            bands[bkey] = dn_to_reflectance(dn, scale=info["scale"], offset=info["offset"])

    cube = stack_to_contract(bands)          # 계약 검증 포함
    vfrac = valid_fraction(cube)
    meta = {
        "scene_id": item.get("id"), "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "nodata_pct": props.get("s2:nodata_pixel_percentage"),
        "valid_fraction": round(vfrac, 4), "proj_epsg": utm,
        "bbox_4326": list(bbox), "gsd_m": TARGET_GSD_M,
        "shape": list(cube.shape), "collection": collection,
        "bands": list(contract.BAND_ORDER),
    }
    if out_npy:
        np.save(out_npy, cube)
        Path(out_npy).with_suffix(".provenance.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
