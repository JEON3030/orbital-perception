#!/usr/bin/env python3
"""전력 대결 — 같은 세그 모델을 CPU vs GPU 로, 20m 픽셀 크기를 쓸어가며
mJ/frame(total·dynamic)·처리시간·throughput 을 실측하고 교차점을 찾는다.

meter.measure 는 프레임마다 새 프로세스를 띄우는 계약(형준 외부프로그램)이라
GPU CUDA 초기화가 추론을 압도한다. 여기선 powerlog.PowerMeter 로 in-process
추론 루프를 감싸 '정상상태 추론 에너지'를 공정하게 잰다(워밍업 제외).

산출: outputs/power_duel/results.json  (그래프·PPT 가 읽음)
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from merge import train_seg
from powerlog import PowerMeter, measure_idle

OUT = Path("outputs/power_duel")
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "runs/sat_seg/best.pt"

# 20m 운영점 맥락: S2 풀타일 = 5490² ≈ 30MP. 0.07MP → 16.8MP 까지 쓸어 교차점을 본다.
SIZES = [256, 512, 768, 1024, 1536, 2048, 2560, 3072]
CPU_MAX_SIZE = 1536     # CPU 는 대형서 프레임당 수십초+메모리폭발 → 상한(위는 GPU만·CPU 외삽)
TARGET_SEC = 4.0        # 각 설정을 이 정도 시간 재도록 반복수 자동조정
BUDGET_SEC = 10.0       # 한 설정 본측정 벽시계 상한(느린 CPU 폭주 방지)
MIN_REPS, MAX_REPS = 3, 400


@torch.no_grad()
def bench_one(model, x_dev, device, idle_watt):
    """정상상태 추론을 재서 report dict 를 돌려준다. OOM 이면 예외."""
    cuda = device.type == "cuda"
    # 워밍업 1회(모델로딩·cudnn 오토튠·전력레일 기동 제외) + 1회로 반복수 산정
    model(x_dev).argmax(1)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(x_dev).argmax(1)
    if cuda:
        torch.cuda.synchronize()
    per = max(time.perf_counter() - t0, 1e-4)
    # 반복수: 목표시간·예산시간 둘 다로 상한(느린 CPU 폭주 방지)
    reps = int(np.clip(min(TARGET_SEC, BUDGET_SEC) / per, MIN_REPS, MAX_REPS))

    pm = PowerMeter(interval_ms=100)
    pm.start()
    for _ in range(reps):
        model(x_dev).argmax(1)
    if cuda:
        torch.cuda.synchronize()
    pm.stop()
    r = pm.report(n_frames=reps, idle_watt=idle_watt)
    r["reps"] = reps
    r["sec_per_frame"] = round(per, 5)
    return r


def main():
    dev_cpu = torch.device("cpu")
    dev_gpu = torch.device("cuda") if torch.cuda.is_available() else None
    print(f"CUDA: {torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if dev_gpu else '-'}")

    print("유휴 전력 측정(3s)…")
    idle = measure_idle(3.0)
    print(f"idle = {idle:.3f} W")

    m_cpu, classes = train_seg.load_model(MODEL, dev_cpu)
    m_gpu = train_seg.load_model(MODEL, dev_gpu)[0] if dev_gpu else None

    rows = []
    for s in SIZES:
        mp = s * s / 1e6
        # 계약 6밴드 유효 큐브 (내용 무관 — 텐서 크기만 중요)
        x = np.random.rand(s, s, 6).astype(np.float32)
        xt = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        row = {"size": s, "megapixels": round(mp, 3)}

        # CPU (대형은 프레임당 수십초라 상한 위는 건너뛴다)
        if s <= CPU_MAX_SIZE:
            try:
                r = bench_one(m_cpu, xt.to(dev_cpu), dev_cpu, idle)
                row["cpu"] = r
                print(f"[{s:>4}²={mp:5.2f}MP] CPU  {r['sec_per_frame']*1000:8.1f} ms/f  "
                      f"{r['mJ_per_frame']:9.1f} mJ/f  {r['fps']:6.2f} fps  {r['avg_power_W']:.2f}W", flush=True)
            except RuntimeError as e:
                row["cpu"] = {"error": str(e)[:80]}
                print(f"[{s:>4}²] CPU  실패: {str(e)[:60]}", flush=True)
        else:
            row["cpu"] = {"skipped": f">{CPU_MAX_SIZE}² CPU 생략(수십초/frame)"}
            print(f"[{s:>4}²={mp:5.2f}MP] CPU  생략(상한 {CPU_MAX_SIZE}²)", flush=True)

        # GPU
        if m_gpu is not None:
            try:
                r = bench_one(m_gpu, xt.to(dev_gpu), dev_gpu, idle)
                row["gpu"] = r
                print(f"[{s:>4}²={mp:5.2f}MP] GPU  {r['sec_per_frame']*1000:8.1f} ms/f  "
                      f"{r['mJ_per_frame']:9.1f} mJ/f  {r['fps']:6.2f} fps  {r['avg_power_W']:.2f}W", flush=True)
                del r
            except RuntimeError as e:
                row["gpu"] = {"error": str(e)[:80]}
                print(f"[{s:>4}²] GPU  실패(OOM?): {str(e)[:60]}", flush=True)
                torch.cuda.empty_cache()
        rows.append(row)
        del x, xt
        gc.collect()
        if dev_gpu:
            torch.cuda.empty_cache()
        # 매 행마다 중간저장(중단돼도 부분결과 보존)
        (OUT / "results.json").write_text(json.dumps({"partial": True, "rows": rows},
                                                     ensure_ascii=False, indent=2))

    out = {
        "device_gpu": torch.cuda.get_device_name(0) if dev_gpu else None,
        "device_cpu": "ARM Cortex-A78AE ×6 (Jetson Orin Nano)",
        "idle_watt": round(idle, 3),
        "full_tile_mp": round(5490 * 5490 / 1e6, 1),
        "note_20m": "S2 20m 풀타일 = 5490² ≈ 30.1MP 가 실제 위성 운영점",
        "rows": rows,
    }
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n→ 저장: {OUT/'results.json'}")


if __name__ == "__main__":
    main()
