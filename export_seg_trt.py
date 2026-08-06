#!/usr/bin/env python3
"""SegFormer(시맨틱 세그) → ONNX → TensorRT FP16 엔진 변환기.

YOLO(export_trt.py)는 ultralytics `model.export(format="engine")` 한 방이면 됐지만,
SegFormer 는 HuggingFace transformers 모델이라 경로가 다르다:
  1) torch → ONNX (logits 만 뽑는 얇은 래퍼로 export)
  2) trtexec 로 ONNX → TensorRT FP16 엔진(.engine)
  3) 런타임(seg_trt.py)이 transformers 없이 엔진만 로드해 추론

이 프로젝트의 시맨틱 세그(SegFormer-B0)는 지금까지 PyTorch/transformers 그대로 돌아
탐지(TRT 엔진, ~20FPS) 대비 병목(~7FPS)이었다. FP16 엔진으로 굳혀 실시간에 가깝게
끌어올린다. 입력은 실시간을 위해 고정 512×512(정확도-속도 맞바꿈, 결과 클래스맵은
원본 해상도로 nearest 업샘플 — semantic.py OOM 수정과 동일 철학).

엔진 입력은 [1,3,512,512] fp32, 출력은 logits [1,C,128,128] fp32(C=road19/aerial150).

사용:
  ./export_seg_trt.sh                 # road, aerial 둘 다 FP16
  ./export_seg_trt.sh --domain aerial
  ./export_seg_trt.sh --size 512 --onnx-only   # 엔진 빌드 없이 ONNX 만

결과: segformer_<domain>.engine / .onnx / _labels.json (repo 루트).
엔진은 이 GPU(Orin)·이 TensorRT(10.3) 전용 — 다른 기기로 옮기면 재변환.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# semantic.py 와 동일: 격리 _deps(transformers 4.46) 를 앞에 둔다.
_ROOT = Path(__file__).resolve().parent
_DEPS = _ROOT / "_deps"
if _DEPS.is_dir() and str(_DEPS) not in sys.path:
    sys.path.insert(0, str(_DEPS))

MODEL_IDS = {
    "road":   "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "aerial": "nvidia/segformer-b0-finetuned-ade-512-512",
}
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def export_onnx(domain, size, opset=17):
    import torch
    # transformers 가 torchvision(바이너리 없음) 을 있다고 오판하지 않게 못박는다.
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    from transformers import SegformerForSemanticSegmentation

    mid = MODEL_IDS[domain]
    print(f"\n── [{domain}] {mid} → ONNX ({size}×{size}) ──────────")
    model = SegformerForSemanticSegmentation.from_pretrained(mid).float().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    n_cls = model.config.num_labels

    class SegWrap(torch.nn.Module):
        """SemanticSegmenterOutput 대신 logits 텐서만 반환(ONNX 친화)."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).logits   # [1,C,H/4,W/4]

    wrap = SegWrap(model).eval()
    dummy = torch.randn(1, 3, size, size, dtype=torch.float32)
    onnx_path = _ROOT / f"segformer_{domain}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrap, dummy, str(onnx_path),
            input_names=["pixel_values"], output_names=["logits"],
            opset_version=opset, do_constant_folding=True, dynamic_axes=None)
    # 라벨/클래스수 저장 → 런타임은 transformers 없이 이 json 만 읽는다.
    labels_path = _ROOT / f"segformer_{domain}_labels.json"
    labels_path.write_text(json.dumps(
        {"domain": domain, "model_id": mid, "num_classes": int(n_cls),
         "input_size": int(size), "id2label": id2label},
        ensure_ascii=False, indent=2))
    print(f"  ✓ {onnx_path.name} (C={n_cls}) + {labels_path.name}")
    return onnx_path


def build_engine(domain, onnx_path, workspace_mib=4096):
    engine_path = _ROOT / f"segformer_{domain}.engine"
    print(f"── [{domain}] trtexec FP16 엔진 빌드 → {engine_path.name} ──────")
    cmd = [TRTEXEC, f"--onnx={onnx_path}", "--fp16",
           f"--saveEngine={engine_path}",
           f"--memPoolSize=workspace:{workspace_mib}"]
    t0 = time.time()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ("/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:"
                              + env.get("LD_LIBRARY_PATH", ""))
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0 or not engine_path.exists():
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"[{domain}] trtexec 실패 (code {proc.returncode})")
    sz = engine_path.stat().st_size / 1e6
    print(f"  ✓ {engine_path.name}  ({sz:.1f} MB, 빌드 {dt:.0f}s)")
    return engine_path


def main():
    ap = argparse.ArgumentParser(description="SegFormer → TensorRT FP16 엔진")
    ap.add_argument("--domain", choices=["road", "aerial", "both"], default="both")
    ap.add_argument("--size", type=int, default=512, help="엔진 고정 입력 변(정사각)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--onnx-only", action="store_true", help="ONNX 만, 엔진 빌드 생략")
    args = ap.parse_args()

    domains = ["road", "aerial"] if args.domain == "both" else [args.domain]
    done = []
    for d in domains:
        onnx_path = export_onnx(d, args.size, args.opset)
        if not args.onnx_only:
            eng = build_engine(d, onnx_path)
            done.append(eng)

    print("\n완료.")
    for d in done:
        print(f"  {d}")
    if done:
        print("\n런타임은 엔진이 있으면 자동으로 TRT 를 씁니다 "
              "(semantic.py / webviz.py). PyTorch 폴백 유지.")


if __name__ == "__main__":
    main()
