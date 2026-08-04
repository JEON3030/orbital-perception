#!/usr/bin/env python3
"""시맨틱 세그멘테이션(SegFormer) + 전력계측 — 위성 온보드용.

이 프로젝트는 원래 YOLO 탐지/인스턴스세그(perception.py)만 있었다. 하지만 위성/항공
관점에서 진짜 필요한 것은 **도로·건물·식생·수면·하늘 같은 '장면 전체'의 픽셀 의미**다
(YOLO COCO 에는 그런 클래스가 없다). 그래서 park-hyun-su/syntax_inference 가 가진
시맨틱 세그(SegFormer)를 이 프로젝트의 **에너지 우선(mJ/frame)** 프레임으로 흡수한다.

  - road   : SegFormer-B0 · Cityscapes 19클래스 (차량/도로 관점)
  - aerial : SegFormer-B0 · ADE20K 150클래스 (항공/위성 관점 — 건물·초지·수면·나무)

젯슨(GPU·fp16)과 노트북(CPU) 양쪽에서 돈다 — 장치 결정은 device.py 가 한다.
전력·에너지는 powerlog.PowerMeter 로 실측하고 CUDA 비동기 보정(device.sync())을 한다.

사용:
  ./semantic.sh input/sample.jpg                    # road, 기본
  ./semantic.sh input/sat.png --domain aerial
  ./semantic.sh input/sample.jpg --device cpu --repeat 20
  결과: outputs/<name>_semantic_<domain>.jpg + _energy.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# orbital-perception 전용 격리 의존성(transformers 4.46 + hf_hub 0.26)을 앞에 둔다.
# 공유 venv 의 최신 hf_hub(1.x)는 torch 2.5 용 구버전 transformers 와 안 맞아서,
# 다른 프로젝트를 건드리지 않으려고 여기 _deps 에 따로 깔아 두었다. → README 참고
_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_deps")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

import cv2
import numpy as np

import device
from powerlog import PowerMeter, measure_idle

MODEL_IDS = {
    "road":   "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "aerial": "nvidia/segformer-b0-finetuned-ade-512-512",
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Cityscapes 19클래스 공식 팔레트(RGB). 논문 그림과 색을 맞춰 비교가 되게 한다.
CITYSCAPES_RGB = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)

_model_cache = {}
_palette_cache = {}


def _palette(domain, n):
    """domain 별 색 LUT [n,3] (RGB). road 는 Cityscapes 공식색, 그 외는 결정적 생성."""
    if domain in _palette_cache:
        return _palette_cache[domain]
    if domain == "road" and n <= len(CITYSCAPES_RGB):
        lut = CITYSCAPES_RGB[:n].copy()
    else:
        # ADE20K 등 다클래스: HSV 를 균등 분할한 결정적 팔레트(매 실행 동일)
        idx = np.arange(n)
        hsv = np.stack([(idx * 47 % 180), np.full(n, 200), np.full(n, 230)], 1).astype(np.uint8)
        lut = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0]
    _palette_cache[domain] = lut
    return lut


def get_model(domain):
    """(processor, model, id2label) 캐시 로딩. 현재 device 로 올린다."""
    key = (domain, device.name(), device.use_half())
    if key in _model_cache:
        return _model_cache[key]
    import transformers  # noqa: F401
    # 이 젯슨 venv 엔 torchvision '메타데이터(버전)'만 남아 있고 실제 바이너리는 없다
    # (perception 의 스텁은 NMS 만 제공). 그대로 두면 transformers 가 torchvision 이
    # 있다고 오판해 `torchvision.transforms` 를 import 하려다 죽는다. SegFormer 는
    # torchvision 이 필요 없으므로 여기서 '없음' 으로 못박는다(이미지 유틸 import 전에).
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    mid = MODEL_IDS[domain]
    proc = SegformerImageProcessor.from_pretrained(mid)
    model = SegformerForSemanticSegmentation.from_pretrained(mid)
    model = device.to_model(model)
    out = (proc, model, model.config.id2label)
    _model_cache[key] = out
    return out


def infer_seg(proc, model, bgr):
    """BGR 이미지 → 클래스맵(HxW int64, 원본 해상도). torch 는 함수 안에서만."""
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, return_tensors="pt")
    px = device.to_inputs(inputs["pixel_values"])
    with torch.no_grad():
        logits = model(pixel_values=px).logits            # [1, C, h/4, w/4]
    # 저해상도 로짓에서 먼저 argmax → 클래스맵만 nearest 업샘플.
    # (기존: 로짓을 원본해상도로 bilinear 업샘플 후 argmax → [1,C,H,W] float 텐서가
    #  150클래스·고해상도에서 수 GB로 폭증해 젯슨 공유메모리 OOM 크래시.)
    # 이제 [1,1,H,W]만 만들어 메모리 C배 절감(경계는 약간 블로키하나 OOM 방지).
    small = logits.argmax(1, keepdim=True).float()        # [1, 1, h/4, w/4]
    up = torch.nn.functional.interpolate(small, size=bgr.shape[:2], mode="nearest")
    return up[0, 0].to("cpu").numpy().astype("int64")


def colorize(seg, domain, id2label):
    """클래스맵 → 색 마스크(BGR)."""
    n = (max(id2label) + 1) if id2label else int(seg.max()) + 1
    lut = _palette(domain, max(n, int(seg.max()) + 1))
    rgb = lut[seg.clip(0, len(lut) - 1)]
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def overlay(bgr, seg, domain, id2label, alpha=0.55):
    """원본 위에 색 마스크를 얹되, 원본 휘도로 밝기를 변조해 경계·질감을 남긴다."""
    mask = colorize(seg, domain, id2label).astype(np.float32)
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lum = (0.4 + 0.6 * lum)[:, :, None]        # 너무 어두워지지 않게 하한
    painted = mask * lum
    return (bgr.astype(np.float32) * (1 - alpha) + painted * alpha).clip(0, 255).astype(np.uint8)


def class_summary(seg, id2label, top=8):
    ids, cnts = np.unique(seg, return_counts=True)
    order = np.argsort(-cnts)
    tot = seg.size
    rows = []
    for i in order[:top]:
        cid = int(ids[i])
        pct = cnts[i] * 100.0 / tot
        if pct < 0.5:
            continue
        rows.append((id2label.get(cid, str(cid)), round(pct, 1)))
    return rows


def report_dict(pm, reps, idle_w):
    r = pm.report(n_frames=reps, idle_watt=idle_w)
    return r


def print_report(r, extra=""):
    print("\n── ⚡ 에너지 리포트 (시맨틱 세그) ─────────────────────────")
    if extra:
        print(extra)
    print(f"  프레임      : {r['frames']}장 / {r['seconds']}s  ({r['fps']} FPS)")
    print(f"  평균 전력   : {r['avg_power_W']} W  (피크 {r['peak_power_W']} W)")
    if r.get("idle_power_W") is not None:
        print(f"  유휴 기준선 : {r['idle_power_W']} W → 동적 {r['dynamic_power_W']} W")
    print(f"  총 에너지   : {r['energy_J']} J")
    print(f"  ▶ 프레임당  : {r['mJ_per_frame']} mJ/frame"
          + (f"  (순수추론 {r['dynamic_mJ_per_frame']} mJ)" if r.get('dynamic_mJ_per_frame') is not None else ""))
    print(f"  ▶ 효율      : {r['frames_per_joule']} frames/J")
    print("──────────────────────────────────────────────────────────")


def measure_image(domain, bgr, reps, idle_w):
    """이미지 1장을 reps회 반복추론하며 에너지 측정. (seg, id2label, report) 반환."""
    proc, model, id2label = get_model(domain)
    device.sync()
    seg = infer_seg(proc, model, bgr)          # 워밍업(측정 제외)
    device.sync()
    pm = PowerMeter()
    pm.start()
    for _ in range(max(reps, 1)):
        seg = infer_seg(proc, model, bgr)
    device.sync()                              # CUDA 비동기 보정 — 큐가 다 빈 뒤 정지
    pm.stop()
    return seg, id2label, report_dict(pm, max(reps, 1), idle_w)


def run_image(path, args, idle_w):
    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"[에러] 이미지 못 읽음: {path}")
        return
    seg, id2label, r = measure_image(args.domain, bgr, args.repeat, idle_w)
    out = overlay(bgr, seg, args.domain, id2label)
    op = Path(args.outdir) / f"{path.stem}_semantic_{args.domain}.jpg"
    cv2.imwrite(str(op), out)
    print(f"\n[semantic·{args.domain}] {path.name} → {op}")
    print("  픽셀 점유 클래스:")
    for name, pct in class_summary(seg, id2label):
        print(f"    - {name}: {pct}%")
    print_report(r, extra=f"  (동일 프레임 {max(args.repeat,1)}회 반복추론 · {device.summary()})")
    payload = {"task": "semantic", "domain": args.domain, "model": MODEL_IDS[args.domain],
               "device": device.name(), "half": device.use_half(),
               "input": str(path), "repeat": max(args.repeat, 1),
               "classes": class_summary(seg, id2label), "energy": r}
    ej = Path(args.outdir) / f"{path.stem}_semantic_{args.domain}_energy.json"
    ej.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"  에너지 JSON → {ej}")


def run_video(source, args, idle_w, name):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        print(f"[에러] 영상/카메라 못 엶: {source}")
        return
    proc, model, id2label = get_model(args.domain)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    op = Path(args.outdir) / f"{name}_semantic_{args.domain}.mp4"
    writer = cv2.VideoWriter(str(op), cv2.VideoWriter_fourcc(*"avc1"),
                             max(fps / max(args.stride, 1), 1.0), (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(op), cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(fps / max(args.stride, 1), 1.0), (w, h))
    from collections import Counter
    tally = Counter()
    n, fi, warmed = 0, -1, False
    pm = PowerMeter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % max(args.stride, 1) != 0:
            continue
        if not warmed:
            infer_seg(proc, model, frame)
            device.sync()
            warmed = True
            pm.start()
        seg = infer_seg(proc, model, frame)
        for name_, pct in class_summary(seg, id2label):
            tally[name_] += 1
        writer.write(overlay(frame, seg, args.domain, id2label))
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    device.sync()
    pm.stop()
    cap.release()
    writer.release()
    print(f"\n[semantic·{args.domain}] {name} → {op}  ({n}프레임)")
    print_report(report_dict(pm, max(n, 1), idle_w))


def main():
    ap = argparse.ArgumentParser(description="시맨틱 세그(SegFormer) + 전력계측")
    ap.add_argument("source", help="이미지/영상 파일·폴더, 또는 카메라 인덱스")
    ap.add_argument("--domain", choices=["road", "aerial"], default="road")
    ap.add_argument("--device", choices=list(device.MODES), default="auto")
    ap.add_argument("--repeat", type=int, default=20, help="이미지 반복추론(계측 안정화)")
    ap.add_argument("--stride", type=int, default=2, help="영상 프레임 스킵")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-idle", action="store_true")
    ap.add_argument("--outdir", default=str(Path(__file__).parent / "outputs"))
    args = ap.parse_args()

    device.set_mode(args.device)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    print(f"장치: {device.summary()}")

    idle_w = None
    if not args.no_idle:
        print("유휴 전력 기준선 측정(2s)...")
        idle_w = measure_idle(2.0)
        print(f"  유휴 평균 {idle_w:.2f} W")

    print(f"모델 로딩: {MODEL_IDS[args.domain]} ({args.domain}) ...")
    get_model(args.domain)

    if str(args.source).isdigit():
        run_video(args.source, args, idle_w, name=f"cam{args.source}")
        return
    inp = Path(args.source)
    files = (sorted(p for p in inp.iterdir() if p.suffix.lower() in IMG_EXT | VID_EXT)
             if inp.is_dir() else [inp])
    for f in files:
        if not f.exists():
            print(f"[에러] 파일 없음: {f}")
            continue
        ext = f.suffix.lower()
        if ext in IMG_EXT:
            run_image(f, args, idle_w)
        elif ext in VID_EXT:
            run_video(str(f), args, idle_w, name=f.stem)
    print(f"\n완료. 결과 폴더: {args.outdir}")


if __name__ == "__main__":
    main()
