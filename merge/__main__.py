#!/usr/bin/env python3
"""merge CLI — python -m merge <명령>. 자세한 개요는 merge/__init__.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import acquire
from . import adopt as adopt_mod
from . import contract, meter, score


def _print_contract() -> None:
    print("계약 — 두 프로그램이 만나는 자리 (이 표는 코드에서 나온다)\n")
    print("  호출:  python 내프로그램.py --in {in} --out {out}")
    print(f"  {{in}}   (H, W, {contract.N_BANDS}) float32 .npy   6밴드 반사율 [0,1]")
    print(f"  {{out}}  (H, W)    uint8   .npy   클래스 번호 (0~{contract.MAX_ID})\n")
    print("클래스 번호표 (RNR 약속 ①)")
    print(f"  {'번호':<4}{'key':<10}{'한글':<10}{'RGB':<16}설명")
    for cid, key, ko, rgb, desc in contract.class_table_rows():
        print(f"  {cid:<4}{key:<10}{ko:<10}{rgb:<16}{desc}")
    print("\n밴드 순서 (RNR 계약, NIR=B8A 20m)")
    print("  " + "  ".join(f"{i}:{b}({contract.BAND_S2[b]})"
                           for i, b in enumerate(contract.BAND_ORDER)))
    print(f"\n좌표계 (약속 ②)  내보내기 EPSG:{contract.EXPORT_EPSG} · "
          f"면적 EPSG:{contract.AREA_EPSG}")
    print(f"\n탐지 출력 규격  {{out_det}}  {contract.DETECTION_SCHEMA} (JSON)")
    print("  선박·항공기=OBB(cx,cy,w,h,angle_deg), 재해=폴리곤([[x,y]…]) · "
          "area_px/area_m²(gsd " + f"{contract.DEFAULT_GSD_M:g}m) · 탐지 0건은 정상")
    print(f"  {'번호':<4}{'key':<10}{'한글':<10}{'기하':<9}설명")
    for cid, key, ko, geom, desc in contract.det_class_table_rows():
        print(f"  {cid:<4}{key:<10}{ko:<10}{geom:<9}{desc}")


def _cmd_acquire(a: argparse.Namespace) -> int:
    try:
        bbox = tuple(float(x) for x in a.bbox.split(","))
        if len(bbox) != 4:
            raise ValueError
    except ValueError:
        print("✗ --bbox 는 minlon,minlat,maxlon,maxlat 네 수여야 한다.")
        return 1
    try:
        meta = acquire.acquire_scene(bbox, a.start, a.end, cloud_max=a.cloud,
                                     collection=a.collection, out_npy=a.out)
    except acquire.AcquireError as e:
        print(f"✗ 취득 중단\n  {e}")
        return 1
    print(f"■ 취득  {meta['scene_id']}  ({meta['datetime']}, 구름 {meta['cloud_cover']}%)")
    print(f"  격자 {tuple(meta['shape'])} @ {meta['gsd_m']:g}m · UTM EPSG:{meta['proj_epsg']} · "
          f"밴드 {meta['bands']}")
    if a.out:
        print(f"\n→ 저장: {a.out} (+ .provenance.json)  — `merge check {a.out} --kind in` 로 확인")
    return 0


def _cmd_check(a: argparse.Namespace) -> int:
    shape = None
    if a.shape:
        h, w = (int(x) for x in a.shape.split(","))
        shape = (h, w)
    if a.kind == "det":
        try:
            doc = contract.load_detection(a.path, image_hw=shape)
        except contract.ContractError as e:
            print(f"✗ 계약 위반\n  {e}")
            return 1
        dets = doc["detections"]
        by: dict[str, int] = {}
        for d in dets:
            ko = contract.DET_BY_ID[d["class_id"]].ko
            by[ko] = by.get(ko, 0) + 1
        summ = ", ".join(f"{k}×{n}" for k, n in by.items()) or "없음"
        print(f"✓ 계약 통과  (det)  탐지 {len(dets)}건 [{summ}]  image_hw={doc['image_hw']}")
        return 0
    arr = np.load(a.path)
    try:
        if a.kind == "in":
            contract.validate_input(arr, path=a.path)
        else:
            contract.validate_output(arr, shape=shape, path=a.path)
    except contract.ContractError as e:
        print(f"✗ 계약 위반\n  {e}")
        return 1
    print(f"✓ 계약 통과  ({a.kind})  shape={tuple(arr.shape)} dtype={arr.dtype}")
    return 0


def _cmd_idle(a: argparse.Namespace) -> int:
    w = meter.measure_idle(a.seconds)
    print(f"유휴 전력 ≈ {w:.3f} W  (이 값을 measure --idle-watt 에 넣으면 dynamic mJ/frame 이 나온다)")
    return 0


def _cmd_measure(a: argparse.Namespace) -> int:
    idle = a.idle_watt
    if a.measure_idle and idle is None:
        idle = meter.measure_idle(a.idle_seconds)
        print(f"[idle] {idle:.3f} W 측정됨\n")
    try:
        res = meter.measure(
            name=a.name, seg_cmd=a.seg_cmd, input_npy=a.input, reps=a.repeat,
            idle_watt=idle, require_gpu=a.require_gpu, min_free_mb=a.min_free_mb,
            out_json=a.out,
        )
    except (RuntimeError, contract.ContractError) as e:
        print(f"✗ 측정 중단\n  {e}")
        return 1
    _print_measure(res)
    if a.out:
        print(f"\n→ 저장: {a.out}")
    return 0


def _print_measure(res) -> None:
    e = res.energy
    ta = res.table_a
    print(f"■ {res.name}  ({ta.get('board')}, {ta.get('power_mode')}, "
          f"{ta.get('scene_px')}px, {res.n_frames}회)")
    print(f"  장당 초(중앙): {sorted(res.per_rep_sec)[len(res.per_rep_sec)//2]:.3f}  "
          f"회차: {res.per_rep_sec}")
    if e:
        print(f"  fps {e.get('fps')} · 총 {e.get('mJ_per_frame')} mJ/frame"
              + (f" · 순수(dynamic) {e.get('dynamic_mJ_per_frame')} mJ/frame"
                 if e.get('dynamic_mJ_per_frame') is not None else " · 순수: 판정 불가(idle 미측정)"))
        print(f"  평균 {e.get('avg_power_W')} W · 피크 {e.get('peak_power_W')} W · "
              f"에너지 {e.get('energy_J')} J")
    print(f"  계약 {'OK' if res.contract_ok else '위반'} · "
          f"메모리 free {ta.get('mem_free_mb')}MB / available {ta.get('mem_available_mb')}MB · "
          f"장치 {ta.get('device_env')}")
    if res.throttle_warn:
        print(f"  ⚠ 열 스로틀 의심 (뒤가 앞의 {res.throttle_ratio}배)")
    for n in res.notes:
        print(f"  · {n}")


def _cmd_score(a: argparse.Namespace) -> int:
    try:
        res = score.score_files(a.pred, a.label, ignore_index=a.ignore)
    except score.ScoreError as e:
        print(f"✗ 정확도 측정 불가\n  {e}")
        return 1
    _print_score(res, a.name)
    if a.out:
        # adopt 가 그대로 읽을 수 있게 accuracy.miou 로 싸서 저장한다(손입력 대체).
        payload = {"name": a.name, "accuracy": {"miou": res["miou"]}, "detail": res}
        Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n→ 저장: {a.out}  (adopt --score-a/--score-b 로 연동)")
    return 0


def _print_score(res: dict, name: str) -> None:
    m = res["miou"]
    print(f"■ 정확도 {name}  (라벨 등장 클래스 {res['n_classes_present']}개, "
          f"유효 {res['n_valid_px']:,}px)")
    print(f"  mIoU {m if m is not None else '판정 불가(유효화소 없음)'} · "
          f"픽셀정확도 {res['pixel_acc']}")
    if res["per_class_iou"]:
        print("  클래스별 IoU:")
        for nm, v in sorted(res["per_class_iou"].items(), key=lambda kv: -kv[1]):
            print(f"    {nm:<10} {v:.3f}")
    if res["worst_classes"]:
        worst = ", ".join(f"{k} {v:.3f}" for k, v in res["worst_classes"])
        print(f"  ⚠ 약한 클래스(먼저 볼 곳): {worst}")


def _load_json(p: str) -> dict:
    return json.loads(Path(p).read_text())


def _merge_score(res: dict, score_path: str) -> None:
    """score 서브커맨드가 낸 score.json 의 실측 mIoU 를 결과에 심는다(손입력 대체)."""
    sj = _load_json(score_path)
    miou = sj.get("accuracy", {}).get("miou")
    if miou is None:
        print(f"[warn] {score_path} 에 accuracy.miou 가 없다 — 정확도는 판정 불가로 남는다.")
    res.setdefault("accuracy", {})["miou"] = miou


def _inject(res: dict, miou, mem, fps) -> dict:
    acc = res.setdefault("accuracy", {})
    if miou is not None:
        acc["miou"] = miou
    if mem is not None:
        acc["peak_mem_mb"] = mem
    if fps is not None:
        acc["display_fps"] = fps
    return res


def _cmd_adopt(a: argparse.Namespace) -> int:
    ra, rb = _load_json(a.a), _load_json(a.b)
    # 실측 정확도(score.json)를 먼저 심고, --miou 손입력이 있으면 그게 덮어쓴다(명시 우선).
    if a.score_a:
        _merge_score(ra, a.score_a)
    if a.score_b:
        _merge_score(rb, a.score_b)
    _inject(ra, a.miou_a, a.mem_a, a.fps_a)
    _inject(rb, a.miou_b, a.mem_b, a.fps_b)
    try:
        out = adopt_mod.adopt(ra, rb, name_a=a.name_a, name_b=a.name_b, incumbent=a.incumbent)
    except (adopt_mod.InvalidComparison, ValueError) as e:
        print(f"✗ 채택표를 낼 수 없다\n  {e}")
        return 1
    _print_adopt(out)
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n→ 저장: {a.out}")
    return 0


def _print_adopt(out: dict) -> None:
    na, nb = out["name_a"], out["name_b"]
    print(f"항목별 채택표 (동률 기준 {int(out['tie_tol']*100)}%, 기존={out['incumbent']})\n")
    print(f"  {'항목':<14}{na:>12}{nb:>12}   {'판정':<10}{'채택':<10}몫")
    for it in out["items"]:
        va = _fmt(it["va"]); vb = _fmt(it["vb"])
        print(f"  {it['label']:<14}{va:>12}{vb:>12}   {it['raw_verdict']:<10}"
              f"{it['winner']:<10}{it['who']}")
    if out["undecided"]:
        print(f"\n  판정 불가(0 으로 안 채움): {', '.join(out['undecided'])}")
    print(f"\n  채택: " + (", ".join(f"{k}={v}" for k, v in out["adopted"].items()) or "(없음)"))
    for w in out["warnings"]:
        print(f"  ⚠ {w}")
    print(f"\n  {out['baseline_note']}")


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _cmd_selftest(a: argparse.Namespace) -> int:
    return _selftest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="merge", description="계약·계측·채택 도구")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("contract", help="계약(클래스표·밴드·규격)을 코드에서 출력")

    q = sub.add_parser("acquire", help="S2 20m 장면을 받아 계약 6밴드 npy 로 (STAC+GDAL)")
    q.add_argument("--bbox", required=True, help="minlon,minlat,maxlon,maxlat (EPSG:4326)")
    q.add_argument("--start", required=True, help="시작일 YYYY-MM-DD")
    q.add_argument("--end", required=True, help="종료일 YYYY-MM-DD")
    q.add_argument("--cloud", type=float, default=20.0, help="구름 상한 %% (기본 20)")
    q.add_argument("--collection", default=acquire.COLLECTION)
    q.add_argument("--out", help="6밴드 float32 npy 저장 경로(+.provenance.json)")

    c = sub.add_parser("check", help="npy(분할)·JSON(탐지)이 계약에 맞는지 검사")
    c.add_argument("path")
    c.add_argument("--shape", help="예상 크기 H,W (출력·탐지 image_hw 대조용)")
    c.add_argument("--kind", choices=["in", "out", "det"], default="out",
                   help="in/out=분할 npy, det=탐지 JSON({out_det})")

    i = sub.add_parser("idle", help="유휴 전력(W) 측정")
    i.add_argument("--seconds", type=float, default=3.0)

    m = sub.add_parser("measure", help="seg-cmd 를 반복 실행하며 total/dynamic 에너지 측정")
    m.add_argument("--name", default="측정")
    m.add_argument("--seg-cmd", required=True, help='"python 내프로그램.py --in {in} --out {out}"')
    m.add_argument("--in", dest="input", required=True, help="(H,W,6) float32 .npy")
    m.add_argument("--repeat", type=int, default=meter.DEFAULT_REPEAT)
    m.add_argument("--idle-watt", type=float, default=None)
    m.add_argument("--measure-idle", action="store_true", help="측정 전 idle 을 자동으로 잰다")
    m.add_argument("--idle-seconds", type=float, default=3.0)
    m.add_argument("--require-gpu", action="store_true", help="CUDA 없으면 멈춤(조용한 CPU 강등 금지)")
    m.add_argument("--min-free-mb", type=int, default=meter.DEFAULT_MIN_FREE_MB)
    m.add_argument("--out", help="결과 JSON 저장 경로")

    s = sub.add_parser("score", help="예측·정답 npy 로 mIoU 실측(형준 정확도 축)")
    s.add_argument("pred", help="예측 (H,W) uint8 .npy — 계약 준수")
    s.add_argument("label", help="정답 (H,W) 정수 .npy — 무효화소 허용")
    s.add_argument("--name", default="측정")
    s.add_argument("--ignore", type=int, default=score.DEFAULT_IGNORE,
                   help=f"라벨 무효화소 값(기본 {score.DEFAULT_IGNORE}, -1 도 자동 무효)")
    s.add_argument("--out", help="score.json 저장(adopt --score-a/-b 로 연동)")

    d = sub.add_parser("adopt", help="두 결과 JSON 으로 항목별 채택표")
    d.add_argument("a"); d.add_argument("b")
    d.add_argument("--name-a", default="A"); d.add_argument("--name-b", default="B")
    d.add_argument("--incumbent", help="먼저 있던 쪽 이름(동률 시 유지)")
    for side in ("a", "b"):
        d.add_argument(f"--score-{side}", default=None, dest=f"score_{side}",
                       help="score 서브커맨드가 낸 score.json(실측 mIoU, 손입력 대체)")
        d.add_argument(f"--miou-{side}", type=float, default=None, dest=f"miou_{side}",
                       help="정확도 손입력(있으면 --score 를 덮어씀 — 명시 우선)")
        d.add_argument(f"--mem-{side}", type=float, default=None, dest=f"mem_{side}")
        d.add_argument(f"--fps-{side}", type=float, default=None, dest=f"fps_{side}")
    d.add_argument("--out")

    sub.add_parser("selftest", help="전 과정·수식 자체검증(젯슨 없이도 됨)")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "contract":
        _print_contract(); return 0
    if a.cmd == "acquire":
        return _cmd_acquire(a)
    if a.cmd == "check":
        return _cmd_check(a)
    if a.cmd == "idle":
        return _cmd_idle(a)
    if a.cmd == "measure":
        return _cmd_measure(a)
    if a.cmd == "score":
        return _cmd_score(a)
    if a.cmd == "adopt":
        return _cmd_adopt(a)
    if a.cmd == "selftest":
        return _cmd_selftest(a)
    return 2


# ── selftest : 젯슨 없이 계약·채택 수식을 검증 ─────────────────────────────
def _selftest() -> int:
    import tempfile
    ok = True

    def expect(cond, msg):
        nonlocal ok
        print(("  ✓ " if cond else "  ✗ ") + msg)
        ok = ok and cond

    print("계약 검사")
    good_in = np.zeros((8, 8, 6), dtype=np.float32)
    expect(contract.validate_input(good_in) is good_in, "정상 입력 통과")
    for bad, why in [
        (np.zeros((8, 8, 3), np.float32), "밴드 3개"),
        (np.zeros((8, 8, 6), np.float64), "float64"),
        (np.full((8, 8, 6), 255, np.float32), "0~255 원자료"),
    ]:
        try:
            contract.validate_input(bad); expect(False, f"불량 입력 거부({why})")
        except contract.ContractError:
            expect(True, f"불량 입력 거부({why})")

    good_out = np.zeros((8, 8), dtype=np.uint8)
    expect(contract.validate_output(good_out, shape=(8, 8)) is good_out, "정상 출력 통과")
    for bad, why in [
        (np.zeros((8, 8), np.int64), "int64"),
        (np.full((8, 8), 42, np.uint8), "클래스 42"),
    ]:
        try:
            contract.validate_output(bad, shape=(8, 8)); expect(False, f"불량 출력 거부({why})")
        except contract.ContractError:
            expect(True, f"불량 출력 거부({why})")

    print("채택 수식 검사")
    # 조건 동일한 두 결과를 손으로 만들어 규칙을 검증한다(계측 없이).
    base_ta = dict(board="X", power_mode="7W", scene_px=2048, mem_free_mb=3000,
                   mem_available_mb=4000, repeat=3, device_env="cpu")
    A = dict(table_a=base_ta, contract_ok=True, per_rep_sec=[1.0, 1.0, 1.0],
             throttle_warn=False, energy={"mJ_per_frame": 100, "dynamic_mJ_per_frame": 40, "fps": 1.0},
             accuracy={"miou": 0.60})
    B = dict(table_a=base_ta, contract_ok=True, per_rep_sec=[2.0, 2.0, 2.0],
             throttle_warn=False, energy={"mJ_per_frame": 50, "dynamic_mJ_per_frame": 20, "fps": 0.5},
             accuracy={"miou": 0.605})    # 0.60 vs 0.605 → 5% 안쪽 → 동률
    out = adopt_mod.adopt(A, B, name_a="우리", name_b="형준", incumbent="우리")
    by = {it["key"]: it for it in out["items"]}
    expect(by["accuracy"]["raw_verdict"] == "동률" and by["accuracy"]["winner"] == "우리",
           "정확도 0.60 vs 0.605 → 동률 → 기존(우리) 유지")
    expect(by["speed"]["winner"] == "우리", "속도 1.0s < 2.0s → 우리")
    expect(by["energy"]["winner"] == "형준", "전기 20 < 40 mJ → 형준")
    expect(by["memory"]["winner"] == adopt_mod.UNDECIDED, "메모리 미측정 → 판정 불가(0 아님)")

    # 표 A 불일치는 예외
    Bbad = dict(B); Bbad["table_a"] = dict(base_ta, power_mode="15W")
    try:
        adopt_mod.adopt(A, Bbad, name_a="우리", name_b="형준", incumbent="우리")
        expect(False, "전력 모드 다르면 비교 거부")
    except adopt_mod.InvalidComparison:
        expect(True, "전력 모드 다르면 비교 거부")

    print("정확도 실측 검사 (mIoU — 형준 축)")
    perfect = np.array([[0, 1], [2, 3]], np.uint8)
    expect(score.score_arrays(perfect, perfect.copy())["miou"] == 1.0,
           "완전 일치 → mIoU 1.0")
    ph = np.array([[0, 0], [1, 1]], np.uint8)
    lh = np.array([[0, 1], [0, 1]], np.uint8)
    expect(abs(score.score_arrays(ph, lh)["miou"] - 1 / 3) < 1e-3,
           "절반 오분류(2클래스) → mIoU 1/3")
    li = np.array([[0, 255], [255, 1]], np.uint8)
    expect(score.score_arrays(ph, li, ignore_index=255)["n_valid_px"] == 2,
           "무효화소(255) 제외 → 유효 2px")
    try:
        score.score_arrays(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8))
        expect(False, "예측·라벨 크기 다르면 측정 거부")
    except score.ScoreError:
        expect(True, "예측·라벨 크기 다르면 측정 거부")

    print("탐지 계약 검사 (seg-파생 탐지: OBB+폴리곤)")
    import copy
    good_det = {
        "schema": contract.DETECTION_SCHEMA, "image_hw": [256, 256], "gsd_m": 20.0,
        "detections": [
            {"class_id": 10, "geom_type": "obb",
             "obb": {"cx": 100.0, "cy": 120.0, "w": 15.0, "h": 4.0, "angle_deg": 30.0},
             "area_px": 55.0, "score": 0.9},
            {"class_id": 12, "geom_type": "polygon",
             "polygon": [[10, 10], [40, 12], [35, 50], [8, 45]], "area_px": 900.0, "score": 0.8},
        ],
    }
    expect(contract.validate_detection(good_det) is good_det, "정상 탐지(선박 OBB+산불 폴리곤) 통과")
    expect(contract.validate_detection(
        {"schema": contract.DETECTION_SCHEMA, "image_hw": [64, 64], "detections": []}) is not None,
        "탐지 0건([]) 정상('선박 없음'은 오류 아님)")
    for mut, why in [
        (lambda d: d.update(schema="geojson"), "schema 틀림"),
        (lambda d: d["detections"][0].update(class_id=5), "분할번호(5) 오용"),
        (lambda d: d["detections"][0]["obb"].update(w=0), "OBB w=0"),
        (lambda d: d["detections"][1].update(polygon=[[1, 1], [2, 2]]), "폴리곤 2점"),
        (lambda d: d["detections"][0]["obb"].update(cx=9999), "좌표 이미지 밖"),
        (lambda d: d["detections"][0].update(score=1.5), "score>1"),
    ]:
        bad = copy.deepcopy(good_det); mut(bad)
        try:
            contract.validate_detection(bad); expect(False, f"불량 탐지 거부({why})")
        except contract.ContractError:
            expect(True, f"불량 탐지 거부({why})")

    print("취득 순수코어 검사 (S2 20m — 네트워크 없이)")
    r = acquire.dn_to_reflectance(np.array([[0, 10000]], np.uint16), scale=0.0001, offset=-0.1)
    expect(r[0, 0] == 0.0 and abs(r[0, 1] - 0.9) < 1e-6, "DN→반사율 scale/offset+clip(0)")
    cube = acquire.stack_to_contract({b: np.full((3, 3), 0.2, np.float32)
                                      for b in contract.BAND_ORDER})
    expect(cube.shape == (3, 3, contract.N_BANDS), "계약 밴드순 스택 (H,W,6)")
    expect(acquire.pick_least_cloudy(
        [{"id": "hi", "properties": {"eo:cloud_cover": 40}},
         {"id": "lo", "properties": {"eo:cloud_cover": 2}}])["id"] == "lo", "최소구름 장면 선택")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "o.npy"; np.save(p, good_out)
        expect(contract.load_output(str(p), shape=(8, 8)) is not None, "npy 왕복 로드")
        dp = Path(td) / "d.json"; dp.write_text(json.dumps(good_det))
        expect(contract.load_detection(str(dp), image_hw=(256, 256)) is not None, "탐지 JSON 왕복 로드")

    print("\n" + ("전부 통과 ✓" if ok else "실패 있음 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
