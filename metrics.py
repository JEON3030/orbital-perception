#!/usr/bin/env python3
"""정확도(mIoU) + 통합 지표 mIoU-per-Joule.

두 프로젝트의 정체성을 합친다:
  - park-hyun-su/syntax_inference : "얼마나 정확한가" → mIoU
  - 이 프로젝트(orbital-perception): "얼마나 적은 전력으로 도는가" → mJ/frame

위성 온보드에서 진짜 물음은 "가장 정확한 모델을 쓸 수 있는가"가 아니라
**"전력 예산 안에서 정확도를 얼마나 사는가"** 이다. 그래서 정확도와 에너지를
하나로 묶은 지표를 정의한다:

    mIoU-per-Joule = mIoU × frames_per_joule = mIoU / (프레임당 에너지[J])

  → 같은 mIoU면 에너지가 적을수록, 같은 에너지면 mIoU가 높을수록 커진다.
    TensorRT·해상도·백본 선택이 이 한 값으로 비교된다.

mIoU 계산은 혼동행렬 기반(넓은 클래스가 점수를 지배하지 않게 클래스별 IoU 평균).
정답 라벨이 필요하므로 eval 은 (이미지, 라벨마스크) 폴더 쌍을 받는다.
"""
from __future__ import annotations

import numpy as np


class ConfusionMatrix:
    """클래스 n개의 혼동행렬 누적기. mat[label, pred]."""

    def __init__(self, n: int, ignore_index: int = 255):
        self.n = int(n)
        self.ignore = ignore_index
        self.mat = np.zeros((self.n, self.n), dtype=np.int64)

    def update(self, pred: np.ndarray, label: np.ndarray) -> None:
        pred = np.asarray(pred).reshape(-1)
        label = np.asarray(label).reshape(-1)
        keep = (label >= 0) & (label < self.n) & (label != self.ignore) & (pred < self.n) & (pred >= 0)
        idx = self.n * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
        self.mat += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        denom = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)
        return iou

    def present(self) -> np.ndarray:
        """라벨에 실제로 등장한 클래스(행 합 > 0)만 True."""
        return self.mat.sum(1) > 0

    def miou(self) -> float:
        iou = self.per_class_iou()
        p = self.present()
        vals = iou[p]
        vals = vals[~np.isnan(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    def pixel_acc(self) -> float:
        tot = self.mat.sum()
        return float(np.diag(self.mat).sum() / tot) if tot else float("nan")


def miou_per_joule(miou: float, mj_per_frame: float) -> float | None:
    """통합 지표. mj_per_frame(mJ) 를 J 로 바꿔 mIoU / (J/frame)."""
    if miou is None or mj_per_frame is None or mj_per_frame <= 0 or np.isnan(miou):
        return None
    j_per_frame = mj_per_frame / 1000.0
    return round(miou / j_per_frame, 4)


def summarize(cm: ConfusionMatrix, id2label: dict | None = None, top_worst: int = 5) -> dict:
    iou = cm.per_class_iou()
    p = cm.present()
    out = {"mIoU": round(cm.miou(), 4), "pixel_acc": round(cm.pixel_acc(), 4)}
    per = {}
    for cid in np.where(p)[0]:
        v = iou[cid]
        if np.isnan(v):
            continue
        name = id2label.get(int(cid), str(int(cid))) if id2label else str(int(cid))
        per[name] = round(float(v), 4)
    out["per_class_iou"] = per
    out["worst_classes"] = sorted(per.items(), key=lambda kv: kv[1])[:top_worst]
    return out


# ── 라벨 폴더 평가 (정확도 + 에너지 동시 측정) ───────────────────────────────
def eval_folder(images_dir, labels_dir, domain="road", n_classes=None,
                remap=None, ignore_index=255, device_mode="auto", limit=0):
    """(이미지, 라벨PNG) 쌍 폴더를 추론하며 mIoU 와 에너지를 함께 잰다.

    - labels_dir 의 각 파일은 images_dir 의 같은 stem 과 짝. 라벨은 클래스 id 가
      픽셀값인 단일채널 PNG.
    - remap: 모델 출력 클래스 id → 평가 클래스 id 딕셔너리(다르면). 없으면 항등.
    반환: {"accuracy": {...}, "energy": {...}, "miou_per_joule": ...}
    """
    from pathlib import Path
    import cv2
    import device
    import semantic
    from powerlog import PowerMeter, measure_idle

    device.set_mode(device_mode)
    proc, model, id2label = semantic.get_model(domain)
    if n_classes is None:
        n_classes = (max(id2label) + 1) if id2label else 19
    remap_lut = None
    if remap:
        remap_lut = np.arange(max(n_classes, max(remap) + 1), dtype=np.int64)
        for k, v in remap.items():
            remap_lut[int(k)] = int(v)

    imgs = sorted(p for p in Path(images_dir).iterdir()
                  if p.suffix.lower() in semantic.IMG_EXT)
    if limit:
        imgs = imgs[:limit]
    cm = ConfusionMatrix(n_classes, ignore_index)

    idle_w = measure_idle(2.0)
    device.sync()
    pm = PowerMeter()
    pm.start()
    n = 0
    for ip in imgs:
        lp = Path(labels_dir) / (ip.stem + ".png")
        if not lp.exists():
            continue
        bgr = cv2.imread(str(ip))
        label = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED)
        if bgr is None or label is None:
            continue
        if label.ndim == 3:
            label = label[:, :, 0]
        seg = semantic.infer_seg(proc, model, bgr)
        if remap_lut is not None:
            seg = remap_lut[seg.clip(0, len(remap_lut) - 1)]
        if label.shape != seg.shape:
            label = cv2.resize(label, (seg.shape[1], seg.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        cm.update(seg, label)
        n += 1
    device.sync()
    pm.stop()

    energy = pm.report(n_frames=max(n, 1), idle_watt=idle_w)
    acc = summarize(cm, id2label)
    return {"images": n, "domain": domain, "device": device.name(),
            "accuracy": acc, "energy": energy,
            "miou_per_joule": miou_per_joule(acc["mIoU"], energy.get("mJ_per_frame")),
            "dynamic_miou_per_joule": miou_per_joule(acc["mIoU"], energy.get("dynamic_mJ_per_frame"))}


def _selftest():
    """정답 라벨 없이 지표 수식만 검증(합성 데이터)."""
    # 1) 완전 일치 → mIoU 1.0
    a = np.array([[0, 1], [2, 3]])
    cm = ConfusionMatrix(4)
    cm.update(a, a)
    assert abs(cm.miou() - 1.0) < 1e-9, cm.miou()
    # 2) 2클래스 절반 오분류 → 알려진 IoU
    pred = np.array([0, 0, 1, 1])
    lab = np.array([0, 1, 0, 1])
    cm2 = ConfusionMatrix(2)
    cm2.update(pred, lab)
    # 각 클래스 tp=1, fp=1, fn=1 → IoU=1/3, mIoU=1/3
    assert abs(cm2.miou() - 1 / 3) < 1e-9, cm2.miou()
    # 3) 통합 지표: mIoU 0.65, 927.6 mJ/frame → 0.65/0.9276
    v = miou_per_joule(0.65, 927.6)
    assert abs(v - 0.7008) < 1e-3, v
    # dynamic 222 mJ → 0.65/0.222
    assert abs(miou_per_joule(0.65, 222.0) - 2.9279) < 1e-3
    print("metrics selftest OK — mIoU 수식·통합지표 검증 통과")
    print(f"  예) mIoU 0.65 @ 927.6 mJ/frame → mIoU/J = {miou_per_joule(0.65, 927.6)}")
    print(f"      mIoU 0.65 @ 222.0 mJ(순수) → mIoU/J = {miou_per_joule(0.65, 222.0)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="mIoU + mIoU-per-Joule 평가")
    ap.add_argument("--images", help="이미지 폴더")
    ap.add_argument("--labels", help="라벨(PNG, 클래스id 픽셀) 폴더")
    ap.add_argument("--domain", choices=["road", "aerial"], default="road")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--remap", help="모델→평가 클래스 매핑 JSON 파일", default=None)
    ap.add_argument("--selftest", action="store_true", help="라벨 없이 지표 수식만 검증")
    a = ap.parse_args()
    if a.selftest or not (a.images and a.labels):
        _selftest()
    else:
        import json
        remap = json.loads(open(a.remap).read()) if a.remap else None
        res = eval_folder(a.images, a.labels, a.domain, remap=remap,
                          device_mode=a.device, limit=a.limit)
        print(json.dumps(res, ensure_ascii=False, indent=2))
