#!/usr/bin/env python3
"""
YOLO11 .pt → TensorRT .engine 변환기 (위성 온보드 저전력용).

TensorRT FP16/INT8 엔진은 커널 융합·저정밀 연산으로 추론 시간과 **프레임당
에너지(mJ/frame)** 를 크게 줄인다. 이 프로젝트의 핵심 지표가 에너지이므로
detect/segment 가중치를 엔진으로 굳혀 두고 perception.py --model 로 물린다.

사용:
  ./export_trt.sh                       # detect+segment, FP16, imgsz 640
  ./export_trt.sh --half --imgsz 512
  ./export_trt.sh --int8 --task detect  # INT8(캘리브레이션 필요, 느릴 수 있음)

결과: yolo11n.engine / yolo11n-seg.engine (원본 .pt 옆에 생성).
엔진은 이 GPU(Orin)·이 TensorRT 버전 전용 — 다른 기기로 옮기면 재변환 필요.
"""
import argparse
import sys
import time
import types
from pathlib import Path


# perception.py 와 동일한 torchvision 스텁 (NV torch 2.5엔 맞는 torchvision 없음)
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
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.ops"] = ops


_install_torchvision_stub()
from ultralytics import YOLO  # noqa: E402

DEFAULT_PT = {"detect": "yolo11n.pt", "segment": "yolo11n-seg.pt"}


def export_one(pt_path, half, int8, imgsz):
    prec = "INT8" if int8 else ("FP16" if half else "FP32")
    print(f"\n── {pt_path}  →  TensorRT {prec}, imgsz {imgsz} ─────────────")
    model = YOLO(pt_path)
    t0 = time.time()
    out = model.export(format="engine", half=half, int8=int8, imgsz=imgsz,
                       device=0, workspace=4, verbose=False)
    dt = time.time() - t0
    eng = Path(out)
    sz = eng.stat().st_size / 1e6 if eng.exists() else 0
    print(f"  ✓ {eng.name}  ({sz:.1f} MB, 변환 {dt:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description="YOLO11 .pt → TensorRT .engine")
    ap.add_argument("--task", choices=["detect", "segment", "both"], default="both")
    ap.add_argument("--half", action="store_true", help="FP16 (권장, 기본)")
    ap.add_argument("--int8", action="store_true", help="INT8 (캘리브레이션 필요)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default=None, help="특정 .pt 지정(생략시 task 기본값)")
    args = ap.parse_args()

    # 아무 정밀도도 안 주면 FP16 기본
    if not args.half and not args.int8:
        args.half = True

    if args.model:
        targets = [args.model]
    elif args.task == "both":
        targets = [DEFAULT_PT["detect"], DEFAULT_PT["segment"]]
    else:
        targets = [DEFAULT_PT[args.task]]

    done = []
    for pt in targets:
        if not Path(pt).exists():
            print(f"[건너뜀] 파일 없음: {pt}")
            continue
        done.append(export_one(pt, args.half, args.int8, args.imgsz))

    print("\n완료. 생성된 엔진:")
    for d in done:
        print(f"  {d}")
    print("\n사용 예:  ./run.sh input/sample.jpg --model yolo11n.engine")


if __name__ == "__main__":
    main()
