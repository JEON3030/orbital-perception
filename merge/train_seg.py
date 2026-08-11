#!/usr/bin/env python3
"""
merge.train_seg — 6밴드 위성 재해 세그 모델(작은 UNet)을 유사라벨 타일로 학습.

위성 전용 전환 ③의 모델. `dataset` 이 만든 타일(x:(H,W,6)f32, y:유사라벨)을 모아
{0 배경, 1 산불, 2 물} 3-클래스 세그를 학습한다. 출력 마스크를 `vectorize` 가 탐지
문서로 바꾼다(선박·항공기 자리엔 이후 다른 라벨을 끼운다).

정직성:
  · 유사라벨 학습이라 상한은 유사라벨 품질이다(진짜 GT 아님).
  · 이벤트가 분리돼(산불 타일엔 물 라벨 없음) 서로의 양성이 배경으로 섞인다 — 약한
    지도. 그래도 배경 대비 각 재해를 배우기엔 충분한 첫 모델이다.
  · 이 보드는 CUDA 드라이버가 낡아 **CPU 학습**이다(온보드 GPU 에너지는 ⑤에서 별개).

작은 UNet(6→16→32→64) — CPU·소량 타일에 맞춘 경량 구조. torch 만 있으면 된다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import contract

CLASSES = ("background", "fire", "water")   # 0,1,2 — vectorize 시 fire→12, water→13
N_CLASSES = len(CLASSES)
KIND_TO_CLASS = {"burn": 1, "water": 2}


# ── 작은 UNet ────────────────────────────────────────────────────────────
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    """6채널 입력 → n_classes. 인코더 3단(16/32/64), 디코더 대칭. CPU 경량."""

    def __init__(self, in_ch=contract.N_BANDS, n_classes=N_CLASSES, base=16):
        super().__init__()
        b = base
        self.e1 = _block(in_ch, b)
        self.e2 = _block(b, b * 2)
        self.e3 = _block(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = _block(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = _block(b * 2, b)
        self.head = nn.Conv2d(b, n_classes, 1)

    def forward(self, x):
        c1 = self.e1(x)
        c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2))
        x = self.d2(torch.cat([self.u2(c3), c2], 1))
        x = self.d1(torch.cat([self.u1(x), c1], 1))
        return self.head(x)


# ── 데이터 ───────────────────────────────────────────────────────────────
def load_tiles(data_dir: str) -> list[tuple[str, int]]:
    """manifest 로 (타일경로, 재해클래스) 목록을 만든다. 이벤트 kind→클래스."""
    root = Path(data_dir)
    man = json.loads((root / "manifest.json").read_text())
    ev_class = {e["event"]: KIND_TO_CLASS[e["kind"]] for e in man["events"] if not e.get("error")}
    items = []
    for ev, cls in ev_class.items():
        for p in sorted((root / ev).glob("tile_*.npz")):
            items.append((str(p), cls))
    return items


class TileDS(torch.utils.data.Dataset):
    """npz 타일 → (x:6×H×W f32, y:H×W long). 유사라벨(0/1)을 재해클래스로 remap."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        d = np.load(path)
        x = np.ascontiguousarray(d["x"].transpose(2, 0, 1), np.float32)   # HWC→CHW
        y = (d["y"].astype(np.int64) > 0).astype(np.int64) * cls          # 1→cls, 0→bg
        return torch.from_numpy(x), torch.from_numpy(y)


# ── 지표 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def confusion(model, loader, device, n=N_CLASSES):
    model.eval()
    cm = torch.zeros(n, n, dtype=torch.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().reshape(-1)
        t = y.reshape(-1)
        k = (t >= 0) & (t < n)
        cm += torch.bincount(n * t[k] + pred[k], minlength=n * n).reshape(n, n)
    return cm


def iou_from_cm(cm):
    tp = cm.diag().double()
    fp = cm.sum(0).double() - tp
    fn = cm.sum(1).double() - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))
    present = cm.sum(1) > 0
    vals = iou[present]
    vals = vals[~torch.isnan(vals)]
    miou = float(vals.mean()) if len(vals) else float("nan")
    return iou, miou


def train(data_dir: str, out_dir: str, *, epochs=20, batch=2, lr=1e-3,
          val_frac=0.2, base=16, seed=0) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_tiles(data_dir)
    if not items:
        raise RuntimeError(f"{data_dir} 에 학습 타일이 없다 — 먼저 `merge dataset`.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    nval = max(1, int(len(items) * val_frac))
    val_i, tr_i = set(idx[:nval].tolist()), idx[nval:].tolist()
    tr = [items[i] for i in tr_i]
    va = [items[i] for i in sorted(val_i)]

    tl = torch.utils.data.DataLoader(TileDS(tr), batch_size=batch, shuffle=True, num_workers=0)
    vl = torch.utils.data.DataLoader(TileDS(va), batch_size=batch, shuffle=False, num_workers=0)

    model = TinyUNet(base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 클래스 불균형(배경 지배) 완화 — 재해 클래스에 가중.
    weight = torch.tensor([1.0, 3.0, 3.0], device=device)
    lossf = nn.CrossEntropyLoss(weight=weight)

    best_miou, best_state = -1.0, None
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tl:
            opt.zero_grad()
            loss = lossf(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        cm = confusion(model, vl, device)
        iou, miou = iou_from_cm(cm)
        hist.append({"epoch": ep, "train_loss": round(tot / max(len(tr), 1), 4),
                     "val_miou": round(miou, 4),
                     "val_iou": {CLASSES[c]: round(float(iou[c]), 4) for c in range(N_CLASSES)}})
        print(f"  ep{ep:02d}  loss {hist[-1]['train_loss']:.4f}  "
              f"val mIoU {miou:.4f}  fire {float(iou[1]):.3f} water {float(iou[2]):.3f}")
        if miou > best_miou:
            best_miou = miou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": CLASSES, "base": base,
                "in_ch": contract.N_BANDS}, out / "best.pt")
    result = {"device": str(device), "n_train": len(tr), "n_val": len(va),
              "epochs": epochs, "best_val_miou": round(best_miou, 4),
              "classes": list(CLASSES), "history": hist,
              "note": "유사라벨 학습 — 상한은 유사라벨 품질(진짜 GT 아님)."}
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


@torch.no_grad()
def load_model(model_path: str):
    """저장된 체크포인트에서 모델·클래스명을 복원한다(CPU)."""
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    m = TinyUNet(in_ch=ck["in_ch"], n_classes=len(ck["classes"]), base=ck["base"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["classes"]


@torch.no_grad()
def predict_mask(cube: np.ndarray, model_path: str, *, mult: int = 8):
    """계약 6밴드 큐브 → 재해 클래스맵 (H,W) uint8 {0 bg,1 fire,2 water}. 클래스명도 반환.
    UNet 풀링 때문에 H,W 를 mult 배수로 패딩 후 원크기로 자른다."""
    cube = contract.validate_input(cube)
    H, W = cube.shape[:2]
    ph, pw = (-H) % mult, (-W) % mult
    x = np.pad(cube, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m, classes = load_model(model_path)
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32))
    pred = m(t).argmax(1)[0].numpy().astype(np.uint8)
    return pred[:H, :W], classes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("merge.train_seg", description="6밴드 재해 세그 학습(유사라벨)")
    ap.add_argument("--data", required=True, help="merge dataset 출력 폴더")
    ap.add_argument("--out", required=True, help="best.pt·results.json 저장 폴더")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16)
    a = ap.parse_args(argv)
    r = train(a.data, a.out, epochs=a.epochs, batch=a.batch, lr=a.lr, base=a.base)
    print(f"\n최고 val mIoU {r['best_val_miou']} ({r['device']}, train {r['n_train']}/val {r['n_val']}) → {a.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
