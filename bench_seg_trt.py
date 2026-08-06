#!/usr/bin/env python3
"""시맨틱 세그 PyTorch vs TensorRT 엔진 속도·에너지 벤치마크.

같은 이미지를 두 백엔드로 N회 반복추론하며 FPS·mJ/frame(전력 실측)을 비교한다.
탐지는 이미 TRT 엔진(~20FPS)인데 시맨틱만 PyTorch(~7FPS)라 병목이었다 — 이 스크립트로
엔진화 이득을 수치로 확인한다.

사용:  ./webviz.sh 대신 아래처럼(같은 env):
  LD_LIBRARY_PATH=... PYTHONNOUSERSITE=1 python bench_seg_trt.py --domain aerial -n 50
또는:  ./bench_seg_trt.sh --domain aerial
"""
import argparse
import time

import cv2

import device
import semantic
from powerlog import PowerMeter, measure_idle


def bench_backend(backend, bgr, reps, idle_w):
    device.sync()
    backend.infer(bgr)                 # 워밍업(측정 제외)
    device.sync()
    pm = PowerMeter(); pm.start()
    t0 = time.time()
    for _ in range(reps):
        seg = backend.infer(bgr)
    device.sync()
    dt = time.time() - t0
    pm.stop()
    r = pm.report(n_frames=reps, idle_watt=idle_w)
    fps = reps / dt if dt else 0.0
    return seg, fps, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["road", "aerial"], default="aerial")
    ap.add_argument("--image", default="input/sample.jpg")
    ap.add_argument("-n", "--reps", type=int, default=50)
    args = ap.parse_args()

    device.set_mode("auto")
    print(f"장치: {device.summary()}")
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"이미지 못 읽음: {args.image}")
    print(f"이미지: {args.image} {bgr.shape[1]}x{bgr.shape[0]}, {args.reps}회 반복\n")

    print("유휴 전력 기준선(2s)...")
    idle_w = measure_idle(2.0)
    print(f"  유휴 {idle_w:.2f} W\n")

    results = {}

    # 1) PyTorch (TRT 강제 비활성)
    semantic._FORCE_TORCH = True
    tb = semantic.get_backend(args.domain)
    print(f"[PyTorch] {semantic.backend_label(tb)} 측정...")
    _, fps_t, r_t = bench_backend(tb, bgr, args.reps, idle_w)
    results["PyTorch"] = (fps_t, r_t)
    print(f"  {fps_t:.2f} FPS · {r_t['mJ_per_frame']} mJ/frame "
          f"(순수 {r_t.get('dynamic_mJ_per_frame')})\n")

    # 2) TensorRT 엔진
    semantic._FORCE_TORCH = False
    if not semantic.use_trt(args.domain):
        print(f"[TRT] {args.domain} 엔진 없음 — export_seg_trt.py 로 먼저 생성하세요.")
        return
    eb = semantic.get_backend(args.domain)
    print(f"[TensorRT] {semantic.backend_label(eb)} 측정...")
    _, fps_e, r_e = bench_backend(eb, bgr, args.reps, idle_w)
    results["TRT-FP16"] = (fps_e, r_e)
    print(f"  {fps_e:.2f} FPS · {r_e['mJ_per_frame']} mJ/frame "
          f"(순수 {r_e.get('dynamic_mJ_per_frame')})\n")

    # 비교
    print("── 결과 (시맨틱 세그, {}) ─────────────────".format(args.domain))
    print(f"  {'백엔드':10s} {'FPS':>7s} {'mJ/frame':>10s} {'순수mJ':>9s}")
    for k, (fps, r) in results.items():
        print(f"  {k:10s} {fps:7.2f} {r['mJ_per_frame']:10.1f} "
              f"{(r.get('dynamic_mJ_per_frame') or 0):9.1f}")
    if "PyTorch" in results and "TRT-FP16" in results:
        fps_t = results["PyTorch"][0]; fps_e = results["TRT-FP16"][0]
        mj_t = results["PyTorch"][1]["mJ_per_frame"]; mj_e = results["TRT-FP16"][1]["mJ_per_frame"]
        print(f"\n  ▶ 속도 {fps_e/fps_t:.2f}배   에너지 {(mj_e-mj_t)/mj_t*100:+.0f}% "
              f"({mj_t:.0f}→{mj_e:.0f} mJ/frame)")


if __name__ == "__main__":
    main()
