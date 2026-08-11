"""merge 패키지 테스트 — 젯슨/torch 없이도 도는 순수 로직만 검증한다.
(전력 계측은 tegrastats 가 필요하므로 여기선 다루지 않고 selftest/실측으로 확인.)"""
import numpy as np
import pytest

from merge import acquire, adopt, contract, labels, score, vectorize


# ── 계약 ──────────────────────────────────────────────────────────────
def test_class_table_is_0_to_9():
    assert [c.id for c in contract.CLASSES] == list(range(10))
    assert contract.BY_KEY["water"].id == 1
    assert contract.PALETTE.shape == (10, 3)


def test_band_order_nir_is_b8a():
    assert contract.BAND_ORDER[3] == "nir"
    assert contract.BAND_S2["nir"] == "B8A"          # B8(10m) 아님
    assert contract.N_BANDS == 6


def test_validate_input_accepts_good():
    x = np.zeros((4, 4, 6), np.float32)
    assert contract.validate_input(x) is x


@pytest.mark.parametrize("arr", [
    np.zeros((4, 4, 3), np.float32),                 # 밴드 수 틀림
    np.zeros((4, 4, 6), np.float64),                 # dtype 틀림
    np.zeros((4, 6), np.float32),                    # 차원 틀림
    np.full((4, 4, 6), 300.0, np.float32),           # 반사율 범위 밖
])
def test_validate_input_rejects_bad(arr):
    with pytest.raises(contract.ContractError):
        contract.validate_input(arr)


def test_validate_output_accepts_good():
    y = np.zeros((5, 5), np.uint8)
    assert contract.validate_output(y, shape=(5, 5)) is y


@pytest.mark.parametrize("arr,shape", [
    (np.zeros((5, 5), np.int64), (5, 5)),            # int64
    (np.full((5, 5), 11, np.uint8), (5, 5)),         # 클래스 11
    (np.zeros((5, 5), np.uint8), (6, 6)),            # 크기 불일치
    (np.zeros((5, 5, 1), np.uint8), None),           # 차원 틀림
])
def test_validate_output_rejects_bad(arr, shape):
    with pytest.raises(contract.ContractError):
        contract.validate_output(arr, shape=shape)


# ── 채택 ──────────────────────────────────────────────────────────────
def _res(**kw):
    base = dict(
        table_a=dict(board="Orin", power_mode="7W", scene_px=2048, mem_free_mb=3000,
                     mem_available_mb=4000, repeat=3, device_env="cpu"),
        contract_ok=True, per_rep_sec=[1.0, 1.0, 1.0], throttle_warn=False,
        energy={"mJ_per_frame": 100, "dynamic_mJ_per_frame": 40, "fps": 1.0},
        accuracy={},
    )
    base.update(kw)
    return base


def test_tie_keeps_incumbent():
    a = _res(accuracy={"miou": 0.600})
    b = _res(accuracy={"miou": 0.605})               # 5% 안쪽
    out = adopt.adopt(a, b, name_a="우리", name_b="형준", incumbent="우리")
    acc = next(i for i in out["items"] if i["key"] == "accuracy")
    assert acc["raw_verdict"] == adopt.TIE
    assert acc["winner"] == "우리"


def test_clear_win_by_energy():
    a = _res(energy={"mJ_per_frame": 100, "dynamic_mJ_per_frame": 40, "fps": 1.0})
    b = _res(energy={"mJ_per_frame": 50, "dynamic_mJ_per_frame": 20, "fps": 1.0})
    out = adopt.adopt(a, b, name_a="우리", name_b="형준", incumbent="우리")
    energy = next(i for i in out["items"] if i["key"] == "energy")
    assert energy["winner"] == "형준"


def test_unmeasured_is_undecided_not_zero():
    out = adopt.adopt(_res(), _res(), name_a="우리", name_b="형준", incumbent="우리")
    mem = next(i for i in out["items"] if i["key"] == "memory")
    assert mem["winner"] == adopt.UNDECIDED
    assert "메모리(MB)" in out["undecided"]


def test_power_mode_mismatch_is_invalid():
    a = _res()
    b = _res(table_a=dict(a["table_a"], power_mode="15W"))
    with pytest.raises(adopt.InvalidComparison):
        adopt.adopt(a, b, name_a="우리", name_b="형준", incumbent="우리")


def test_scene_size_mismatch_is_invalid():
    a = _res()
    b = _res(table_a=dict(a["table_a"], scene_px=1024))
    with pytest.raises(adopt.InvalidComparison):
        adopt.adopt(a, b, name_a="우리", name_b="형준", incumbent="우리")


def test_low_repeat_warns_not_fatal():
    a = _res(table_a=dict(_res()["table_a"], repeat=1))
    out = adopt.adopt(a, _res(), name_a="우리", name_b="형준", incumbent="우리")
    assert any("반복" in w for w in out["warnings"])


# ── 정확도 실측 (mIoU — 형준 축) ──────────────────────────────────────────
def test_score_perfect_match_is_one():
    a = np.array([[0, 1], [2, 3]], np.uint8)
    r = score.score_arrays(a, a.copy())
    assert r["miou"] == 1.0 and r["pixel_acc"] == 1.0


def test_score_known_half_miou():
    # 2클래스, 각 tp1/fp1/fn1 → IoU 1/3 → mIoU 1/3
    pred = np.array([[0, 0], [1, 1]], np.uint8)
    lab = np.array([[0, 1], [0, 1]], np.uint8)
    assert abs(score.score_arrays(pred, lab)["miou"] - 1 / 3) < 1e-3   # 결과는 소수 4자리 반올림


def test_score_ignore_index_excluded():
    pred = np.array([[0, 0], [1, 1]], np.uint8)
    lab = np.array([[0, 255], [255, 1]], np.uint8)      # 대각선만 유효
    r = score.score_arrays(pred, lab, ignore_index=255)
    assert r["n_valid_px"] == 2 and r["miou"] == 1.0


def test_score_negative_one_is_nodata():
    pred = np.array([[0, 0], [1, 1]], np.uint8)
    lab = np.array([[0, -1], [-1, 1]], np.int64)        # -1 도 무효
    r = score.score_arrays(pred, lab)
    assert r["n_valid_px"] == 2 and r["miou"] == 1.0


def test_score_shape_mismatch_raises():
    with pytest.raises(score.ScoreError):
        score.score_arrays(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8))


def test_score_pred_must_obey_contract():
    # 예측이 int64(계약 위반) → ScoreError 로 올린다(조용히 넘기지 않음)
    with pytest.raises(score.ScoreError):
        score.score_arrays(np.zeros((4, 4), np.int64), np.zeros((4, 4), np.uint8))


def test_score_label_out_of_range_rejected():
    with pytest.raises(score.ScoreError):
        score.score_arrays(np.zeros((3, 3), np.uint8), np.full((3, 3), 12, np.uint8))


def test_score_class_names_from_contract():
    pred = np.array([[1, 1], [2, 2]], np.uint8)          # 1=수체, 2=산림
    per = score.score_arrays(pred, pred.copy())["per_class_iou"]
    assert "수체" in per and "산림" in per


def test_measured_score_feeds_adopt_not_undecided():
    # 실측 mIoU 가 채택표의 정확도 항목을 '판정 불가'가 아니라 실제 승패로 만든다
    pred = np.array([[0, 1], [2, 3]], np.uint8)
    s = score.score_arrays(pred, pred.copy())            # mIoU 1.0
    a = _res(accuracy={"miou": s["miou"]})
    b = _res(accuracy={"miou": 0.5})
    out = adopt.adopt(a, b, name_a="우리", name_b="형준", incumbent="우리")
    acc = next(i for i in out["items"] if i["key"] == "accuracy")
    assert acc["winner"] == "우리" and acc["raw_verdict"] == "A"


# ── 탐지 계약 (seg-파생 탐지: OBB=선박·항공기, 폴리곤=재해) ──────────────────
def _det(**kw):
    base = {
        "schema": contract.DETECTION_SCHEMA, "image_hw": [128, 128], "gsd_m": 20.0,
        "detections": [
            {"class_id": 10, "geom_type": "obb",
             "obb": {"cx": 50.0, "cy": 60.0, "w": 20.0, "h": 5.0, "angle_deg": 15.0},
             "area_px": 80.0, "score": 0.95},
        ],
    }
    base.update(kw)
    return base


def test_detection_accepts_obb_and_polygon():
    d = _det(detections=[
        {"class_id": 10, "geom_type": "obb",
         "obb": {"cx": 30, "cy": 40, "w": 12, "h": 4, "angle_deg": 0}, "area_px": 40, "score": 0.7},
        {"class_id": 13, "geom_type": "polygon",
         "polygon": [[1, 1], [10, 2], [8, 20]], "area_px": 70, "score": 0.6},
    ])
    assert contract.validate_detection(d) is d


def test_detection_empty_list_is_valid():
    assert contract.validate_detection(_det(detections=[])) is not None


def test_detection_class_id_must_be_det_range():
    d = _det(); d["detections"][0]["class_id"] = 2          # 분할(산림) 번호 오용
    with pytest.raises(contract.ContractError):
        contract.validate_detection(d)


def test_detection_obb_needs_positive_size():
    d = _det(); d["detections"][0]["obb"]["h"] = 0
    with pytest.raises(contract.ContractError):
        contract.validate_detection(d)


def test_detection_polygon_needs_three_points():
    d = _det(detections=[{"class_id": 12, "geom_type": "polygon",
                          "polygon": [[0, 0], [1, 1]], "area_px": 5}])
    with pytest.raises(contract.ContractError):
        contract.validate_detection(d)


def test_detection_coords_within_image():
    d = _det(); d["detections"][0]["obb"]["cx"] = 999        # 128px 밖
    with pytest.raises(contract.ContractError):
        contract.validate_detection(d)


def test_detection_wrong_schema_rejected():
    with pytest.raises(contract.ContractError):
        contract.validate_detection(_det(schema="geojson"))


def test_detection_image_hw_cross_check():
    # 분할 출력이 (100,100)인데 탐지 image_hw 가 (128,128)면 거부(다른 장면)
    with pytest.raises(contract.ContractError):
        contract.validate_detection(_det(), image_hw=(100, 100))


def test_detection_score_range():
    d = _det(); d["detections"][0]["score"] = 1.4
    with pytest.raises(contract.ContractError):
        contract.validate_detection(d)


def test_detection_area_m2_helper():
    assert contract.detection_area_m2(10, 20) == 4000.0     # 10px × (20m)²


def test_detection_ids_disjoint_from_seg():
    # 탐지 id 는 분할 0~9 와 절대 안 겹친다(면적표 오염 방지)
    assert set(contract.DET_BY_ID).isdisjoint(set(contract.BY_ID))
    assert min(contract.DET_BY_ID) >= contract.N_CLASSES


# ── 취득 순수 핵심 (S2 20m — 네트워크 없이 검증) ────────────────────────────
def test_acquire_reflectance_scale_offset_and_clip():
    dn = np.array([[0, 10000], [1000, 5000]], np.uint16)
    r = acquire.dn_to_reflectance(dn, scale=0.0001, offset=-0.1)   # baseline 04.00
    assert r.dtype == np.float32
    assert r[0, 0] == 0.0                       # 0*s-0.1 = -0.1 → clip 0
    assert abs(r[0, 1] - 0.9) < 1e-6            # 10000*0.0001-0.1
    assert abs(r[1, 0] - 0.0) < 1e-6            # 1000*0.0001-0.1 = 0


def test_acquire_stack_orders_by_contract_and_validates():
    bands = {b: np.full((4, 4), 0.2, np.float32) for b in contract.BAND_ORDER}
    cube = acquire.stack_to_contract(bands)
    assert cube.shape == (4, 4, contract.N_BANDS) and cube.dtype == np.float32
    contract.validate_input(cube)               # 계약을 통과해야 한다


def test_acquire_stack_missing_band_fails():
    bands = {b: np.zeros((4, 4), np.float32) for b in contract.BAND_ORDER[:-1]}
    with pytest.raises(acquire.AcquireError):
        acquire.stack_to_contract(bands)


def test_acquire_stack_shape_mismatch_fails():
    bands = {b: np.zeros((4, 4), np.float32) for b in contract.BAND_ORDER}
    bands["swir2"] = np.zeros((4, 5), np.float32)
    with pytest.raises(acquire.AcquireError):
        acquire.stack_to_contract(bands)


def test_acquire_pick_least_cloudy():
    items = [{"id": "a", "properties": {"eo:cloud_cover": 40}},
             {"id": "b", "properties": {"eo:cloud_cover": 3}},
             {"id": "c", "properties": {"eo:cloud_cover": 20}}]
    assert acquire.pick_least_cloudy(items)["id"] == "b"


def test_acquire_pick_empty_fails():
    with pytest.raises(acquire.AcquireError):
        acquire.pick_least_cloudy([])


def test_acquire_band_hrefs_maps_nir_to_b8a():
    item = {"assets": {
        akey: {"href": f"https://x/{akey}.tif", "raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
        for akey in acquire.ASSET_KEYS.values()}}
    hrefs = acquire.band_hrefs(item)
    assert set(hrefs) == set(contract.BAND_ORDER)
    assert hrefs["nir"]["href"].endswith("nir08.tif")   # 계약 NIR=B8A(nir08)
    assert hrefs["blue"]["offset"] == -0.1


def test_acquire_band_hrefs_missing_asset_fails():
    item = {"assets": {"blue": {"href": "x"}}}           # 나머지 자산 없음
    with pytest.raises(acquire.AcquireError):
        acquire.band_hrefs(item)


# ── 분광지수 유사라벨 (재해 학습 정답) ──────────────────────────────────────
def _cube(**bands):
    c = np.full((4, 4, contract.N_BANDS), 0.1, np.float32)
    for i, name in enumerate(contract.BAND_ORDER):
        if name in bands:
            c[..., i] = bands[name]
    return c


def test_spectral_indices_use_contract_bands():
    c = _cube(green=0.3, red=0.05, nir=0.1, swir1=0.1, swir2=0.3)
    assert abs(labels.spectral_index(c, "ndwi").mean() - 0.5) < 1e-6      # (g-nir)/(g+nir)
    assert abs(labels.spectral_index(c, "mndwi").mean() - 0.5) < 1e-6     # (g-swir1)/(g+swir1)
    assert abs(labels.spectral_index(c, "ndvi").mean() - 1 / 3) < 1e-5    # (nir-red)/(nir+red)
    assert abs(labels.spectral_index(c, "nbr").mean() - (-0.5)) < 1e-6    # (nir-swir2)/(nir+swir2)


def test_spectral_index_safe_divide():
    c = np.zeros((3, 3, contract.N_BANDS), np.float32)   # 모두 0 → 분모 0 → 0
    assert np.all(labels.spectral_index(c, "ndwi") == 0.0)


def test_pseudo_water_threshold():
    wet = _cube(green=0.3, swir1=0.1)                     # mndwi=0.5>0 → water
    assert labels.coverage(labels.pseudo_water(wet)) == 1.0
    dry = _cube(green=0.1, swir1=0.3)                     # mndwi=-0.5 → not water
    assert labels.coverage(labels.pseudo_water(dry)) == 0.0


def test_pseudo_burn_bitemporal_dnbr():
    pre = _cube(nir=0.6, swir2=0.2)                       # NBR_pre=0.5
    post = _cube(nir=0.1, swir2=0.3)                      # NBR_post=-0.5 → dNBR=1.0≥0.27
    assert labels.coverage(labels.pseudo_burn(post, pre=pre)) == 1.0
    assert labels.coverage(labels.pseudo_burn(post, pre=post)) == 0.0    # dNBR=0


def test_pseudo_burn_monotemporal_approx():
    burn = _cube(red=0.3, nir=0.1, swir2=0.3)             # NBR=-0.5<0, NDVI=(0.1-0.3)/0.4=-0.5<0.2
    assert labels.coverage(labels.pseudo_burn(burn)) == 1.0


def test_pseudo_burn_pre_shape_mismatch_fails():
    post = _cube(nir=0.1, swir2=0.3)
    pre = np.full((3, 3, contract.N_BANDS), 0.2, np.float32)
    with pytest.raises(ValueError):
        labels.pseudo_burn(post, pre=pre)


# ── seg-파생 탐지 벡터화 ────────────────────────────────────────────────────
def _rect_mask(h=20, w=20, r0=5, r1=15, c0=5, c1=15):
    m = np.zeros((h, w), np.uint8); m[r0:r1, c0:c1] = 1
    return m


def test_vectorize_obb_for_ship():
    doc = vectorize.vectorize_mask(_rect_mask(), 10)     # ship → obb
    assert contract.validate_detection(doc) is doc
    assert len(doc["detections"]) == 1
    d = doc["detections"][0]
    assert d["geom_type"] == "obb" and d["obb"]["w"] > 0 and d["area_px"] > 4


def test_vectorize_polygon_for_flood():
    doc = vectorize.vectorize_mask(_rect_mask(), 13)     # flood → polygon
    d = doc["detections"][0]
    assert d["geom_type"] == "polygon" and len(d["polygon"]) >= 3
    assert d["area_m2"] == contract.detection_area_m2(d["area_px"], contract.DEFAULT_GSD_M)


def test_vectorize_drops_small_fragments():
    m = np.zeros((20, 20), np.uint8); m[1, 1] = 1        # 1px → area<4
    doc = vectorize.vectorize_mask(m, 12, min_area_px=4)
    assert len(doc["detections"]) == 0 and doc["vectorize"]["dropped_small"] == 1


def test_vectorize_empty_mask_is_valid_zero():
    doc = vectorize.vectorize_mask(np.zeros((10, 10), np.uint8), 13)
    assert doc["detections"] == [] and contract.validate_detection(doc) is doc


def test_vectorize_two_objects():
    m = np.zeros((30, 30), np.uint8); m[3:9, 3:9] = 1; m[20:27, 20:27] = 1
    doc = vectorize.vectorize_mask(m, 10)
    assert len(doc["detections"]) == 2


def test_vectorize_bad_class_id_rejected():
    with pytest.raises(ValueError):
        vectorize.vectorize_mask(_rect_mask(), 3)        # 분할 번호는 탐지 아님
