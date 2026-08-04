#!/usr/bin/env python3
"""
Orbital Perception — 전력계측 내장 온보드 탐지/세그멘테이션.

최종 목표: 인공위성(소형위성/큐브샛) 탑재. 그래서 지표가 FPS가 아니라
'프레임당 에너지(mJ/frame)'와 'frames-per-joule'다. 이 스크립트는 이미지/영상/
(향후)카메라 입력을 받아 YOLO11로 탐지 또는 세그멘테이션을 수행하면서,
tegrastats로 보드 전력을 실측해 에너지 리포트를 함께 낸다.

작업(task):
  detect   — 사물/사람 박스 + 라벨 (COCO 80종, yolo11n.pt)
  segment  — 인스턴스 세그멘테이션 마스크 + 라벨 (yolo11n-seg.pt)
  (satellite U-Net 세그 훅은 README 참고: ~/wildfire-seg 모델을 --model 로 연결 예정)

입력(source):
  이미지/영상 파일 또는 폴더. 정수(예: 0)면 카메라 /dev/videoN (카메라 연결 시).

사용 예:
  ./run.sh input/photo.jpg
  ./run.sh input/clip.mp4 --task segment
  ./run.sh input/photo.jpg --classes person car --imgsz 512
  ./run.sh 0 --task detect            # 카메라(연결 후)

결과: outputs/<name>_<task>.jpg|mp4  +  outputs/<name>_<task>_energy.json
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path
from collections import Counter

import cv2

from powerlog import PowerMeter, measure_idle


# ── torchvision 스텁 (이 Jetson의 NV torch 2.5엔 맞는 torchvision 바이너리가 없음) ──
# ultralytics는 NMS에만 torchvision을 쓰므로 그 부분만 순수 torch로 끼워넣는다.
def _install_torchvision_stub():
    if "torchvision" in sys.modules:
        return
    import torch

    def nms(boxes, scores, iou_thres):
        keep = []
        order = scores.argsort(descending=True)
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = torch.minimum(boxes[i, 3], boxes[rest, 3])
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-7)
            order = rest[iou <= iou_thres]
        if keep:
            return torch.stack(keep)
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    def batched_nms(boxes, scores, idxs, iou_thres):
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        max_coord = boxes.max()
        offsets = idxs.to(boxes) * (max_coord + 1)
        return nms(boxes + offsets[:, None], scores, iou_thres)

    ops = types.ModuleType("torchvision.ops")
    ops.nms = nms
    ops.batched_nms = batched_nms
    tv = types.ModuleType("torchvision")
    tv.ops = ops
    tv.__version__ = "stub-nms-only"
    # __spec__ 를 붙여 둔다. 안 붙이면 같은 프로세스의 transformers 가
    # importlib.util.find_spec("torchvision") 을 부를 때 "__spec__ is None" 으로
    # 죽는다(SegFormer 시맨틱 세그와 공존). spec 이 있으면 transformers 는
    # 메타데이터 버전 조회에 실패해 "torchvision 없음" 으로 올바로 판정한다.
    import importlib.util as _ilu
    tv.__spec__ = _ilu.spec_from_loader("torchvision", loader=None)
    ops.__spec__ = _ilu.spec_from_loader("torchvision.ops", loader=None)
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.ops"] = ops


_install_torchvision_stub()
from ultralytics import YOLO  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

DEFAULT_MODEL = {"detect": "yolo11n.pt", "segment": "yolo11n-seg.pt"}


def summarize(names, class_ids):
    c = Counter(int(i) for i in class_ids)
    if not c:
        return "  (탐지된 객체 없음)"
    return "\n".join(f"  - {names[k]}: {v}" for k, v in c.most_common())


def infer_once(model, frame, args):
    return model(frame, conf=args.conf, imgsz=args.imgsz, device=args.device,
                 classes=args.classes, retina_masks=(args.task == "segment"),
                 verbose=False)[0]


def infer_track(model, frame, args, persist):
    # ByteTrack: 프레임 간 객체를 연관(추적) → 어려운 프레임에서 놓친 탐지를
    # 트랙이 이어줘 영상 인식률·안정성 상승. persist=False면 트랙 상태 초기화(새 영상 시작).
    return model.track(frame, persist=persist, tracker=args.tracker,
                       conf=args.conf, imgsz=args.imgsz, device=args.device,
                       classes=args.classes, retina_masks=(args.task == "segment"),
                       verbose=False)[0]


def save_energy(args, name, report, extra):
    out = Path(args.outdir) / f"{name}_{args.task}_energy.json"
    payload = {"task": args.task, "model": args.model, "imgsz": args.imgsz,
               "device": str(args.device), **extra, "energy": report}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out


def print_report(report, extra_line=""):
    r = report
    print("\n── ⚡ 에너지 리포트 (위성 전력지표) ──────────────────────")
    if extra_line:
        print(extra_line)
    print(f"  프레임      : {r['frames']}장 / {r['seconds']}s  ({r['fps']} FPS)")
    print(f"  평균 전력   : {r['avg_power_W']} W   (피크 {r['peak_power_W']} W)")
    if "idle_power_W" in r:
        print(f"  유휴 기준선 : {r['idle_power_W']} W  →  동적(추론) {r['dynamic_power_W']} W")
    print(f"  총 에너지   : {r['energy_J']} J")
    print(f"  ▶ 프레임당  : {r['mJ_per_frame']} mJ/frame", end="")
    if r.get("dynamic_mJ_per_frame") is not None:
        print(f"   (순수추론 {r['dynamic_mJ_per_frame']} mJ)")
    else:
        print()
    print(f"  ▶ 효율      : {r['frames_per_joule']} frames/J")
    print("─────────────────────────────────────────────────────────")


def run_image(model, path, args, idle_w):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[에러] 이미지 못 읽음: {path}")
        return
    # 워밍업 1회 (첫 추론은 커널 컴파일 때문에 전력/시간 왜곡 → 계측 제외)
    infer_once(model, img, args)

    reps = max(args.repeat, 1)
    with PowerMeter() as pm:
        res = None
        for _ in range(reps):
            res = infer_once(model, img, args)
    annotated = res.plot()
    out_path = Path(args.outdir) / f"{path.stem}_{args.task}.jpg"
    cv2.imwrite(str(out_path), annotated)

    print(f"\n[{args.task}] {path.name} → {out_path}")
    print("결과:")
    print(summarize(model.names, res.boxes.cls.tolist() if res.boxes is not None else []))
    report = pm.report(n_frames=reps, idle_watt=idle_w)
    print_report(report, extra_line=f"  (동일 프레임 {reps}회 반복 추론 평균)")
    ej = save_energy(args, path.stem, report, {"input": str(path), "repeat": reps})
    print(f"  에너지 JSON → {ej}")


def _open_source(source):
    """파일이면 경로, 정수/‘0’이면 카메라 인덱스로 연다."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    return cap


def run_video(model, source, args, idle_w, name):
    cap = _open_source(source)
    if not cap.isOpened():
        print(f"[에러] 영상/카메라 못 엶: {source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = Path(args.outdir) / f"{name}_{args.task}.mp4"
    # 브라우저 재생 위해 avc1(H.264) 우선, 실패 시 mp4v
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                             fps / max(args.stride, 1), (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / max(args.stride, 1), (w, h))

    try:
        from tqdm import tqdm
        pbar = tqdm(total=(total // max(args.stride, 1)) or None, unit="f",
                    desc=f"{args.task} {name}")
    except Exception:
        pbar = None

    tally = Counter()
    seen_ids = set()                      # 추적 시 고유 객체(트랙ID) 집계
    n = 0
    fi = -1
    warmed = False
    first_track = True                    # 새 영상: 첫 track 호출은 상태 초기화
    pm = PowerMeter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % max(args.stride, 1) != 0:
            continue
        if not warmed:                    # 첫 프레임 워밍업 후 계측 시작(predict, 트랙 미오염)
            infer_once(model, frame, args)
            warmed = True
            pm.start()
        if args.track:
            res = infer_track(model, frame, args, persist=not first_track)
            first_track = False
        else:
            res = infer_once(model, frame, args)
        if res.boxes is not None:
            cls_list = [int(i) for i in res.boxes.cls.tolist()]
            tally.update(cls_list)
            if args.track and res.boxes.id is not None:
                ids = [int(i) for i in res.boxes.id.tolist()]
                seen_ids.update(zip(cls_list, ids))
        writer.write(res.plot())
        n += 1
        if pbar:
            pbar.update(1)
        if args.max_frames and n >= args.max_frames:
            break
    pm.stop()
    if pbar:
        pbar.close()
    cap.release()
    writer.release()

    print(f"\n[{args.task}] {name} → {out_path}")
    print("누적 탐지:")
    print(summarize(model.names, list(tally.elements())))
    if args.track:
        uniq = Counter(c for c, _ in seen_ids)
        print("고유 객체(추적ID 기준):")
        print(summarize(model.names, list(uniq.elements())) if uniq
              else "  (추적된 객체 없음)")
    report = pm.report(n_frames=n, idle_watt=idle_w)
    print_report(report)
    ej = save_energy(args, name, report, {"input": str(source), "stride": args.stride})
    print(f"  에너지 JSON → {ej}")


def main():
    ap = argparse.ArgumentParser(description="Orbital Perception — 전력계측 내장 탐지/세그")
    ap.add_argument("source", help="이미지/영상 파일·폴더, 또는 카메라 인덱스(예: 0)")
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--model", default=None, help="가중치(.pt/.engine). 생략시 task 기본값")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640, help="추론 크기(작을수록 저전력)")
    ap.add_argument("--device", default=0, help="0=GPU, cpu=CPU")
    ap.add_argument("--classes", nargs="*", default=None, help="필터 클래스(person car ...)")
    ap.add_argument("--stride", type=int, default=1, help="영상 프레임 건너뛰기(N중 1장)")
    ap.add_argument("--track", action="store_true",
                    help="영상 시간축 추적(ByteTrack) — 놓친 프레임 보완·깜빡임 제거")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="추적기 설정(bytetrack.yaml/botsort.yaml)")
    ap.add_argument("--max-frames", type=int, default=0, help="영상 최대 프레임(0=전체)")
    ap.add_argument("--repeat", type=int, default=30, help="이미지: 계측용 반복 추론 횟수")
    ap.add_argument("--no-idle", action="store_true", help="유휴전력 사전측정 생략")
    ap.add_argument("--outdir", default=str(Path(__file__).parent / "outputs"))
    args = ap.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODEL[args.task]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 유휴 전력 기준선(순수 추론 에너지 계산용)
    idle_w = None
    if not args.no_idle:
        print("유휴 전력 기준선 측정(2s)...")
        idle_w = measure_idle(2.0)
        print(f"  유휴 평균 {idle_w:.2f} W")

    print(f"모델 로딩: {args.model} ({args.task}) ...")
    model = YOLO(args.model)

    if args.classes:
        name2id = {v: k for k, v in model.names.items()}
        ids = []
        for c in args.classes:
            if c.isdigit():
                ids.append(int(c))
            elif c in name2id:
                ids.append(name2id[c])
            else:
                print(f"[경고] 알 수 없는 클래스 '{c}' 무시")
        args.classes = ids or None
        if args.classes:
            print("필터:", [model.names[i] for i in args.classes])

    # 카메라 인덱스면 바로 영상 경로
    if isinstance(args.source, str) and args.source.isdigit():
        run_video(model, args.source, args, idle_w, name=f"cam{args.source}")
        return

    inp = Path(args.source)
    if inp.is_dir():
        files = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMG_EXT | VID_EXT)
        if not files:
            print(f"[에러] 폴더에 처리할 파일 없음: {inp}")
            sys.exit(1)
    else:
        files = [inp]

    for f in files:
        if not f.exists():
            print(f"[에러] 파일 없음: {f}")
            continue
        ext = f.suffix.lower()
        if ext in IMG_EXT:
            run_image(model, f, args, idle_w)
        elif ext in VID_EXT:
            run_video(model, str(f), args, idle_w, name=f.stem)
        else:
            print(f"[건너뜀] 미지원 형식: {f.name}")

    print(f"\n완료. 결과 폴더: {args.outdir}")


if __name__ == "__main__":
    main()
