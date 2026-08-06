#!/usr/bin/env python3
"""SegFormer TensorRT 엔진 런타임 — transformers 없이 엔진만으로 시맨틱 세그.

export_seg_trt.py 가 만든 `segformer_<domain>.engine` + `_labels.json` 을 로드해
BGR 프레임 → 클래스맵(원본 해상도)을 돌려준다. 인터페이스는 semantic.infer_seg 와
동일(입력 BGR ndarray, 출력 HxW int64)이라 기존 colorize/overlay/class_summary 가
그대로 재사용된다.

CUDA 버퍼는 pycuda 없이 torch 텐서(data_ptr)로 잡는다 — 이 프로젝트 venv 의 NV torch
를 그대로 쓰고 의존성을 안 늘린다. 엔진 입력은 고정 [1,3,S,S] fp32, 출력 logits
[1,C,S/4,S/4]. 저해상도에서 먼저 argmax → 클래스맵만 nearest 업샘플(OOM/속도 절감).
"""
import json
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent
_cache = {}   # domain -> TRTSeg

# ImageNet 정규화(SegformerImageProcessor 기본과 동일)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def engine_path(domain):
    return _ROOT / f"segformer_{domain}.engine"


def labels_path(domain):
    return _ROOT / f"segformer_{domain}_labels.json"


def available(domain):
    """이 도메인의 TRT 엔진+라벨이 준비돼 있으면 True."""
    return engine_path(domain).exists() and labels_path(domain).exists()


def _trt_dtype_to_torch(dt):
    import tensorrt as trt
    import torch
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
    }.get(dt, torch.float32)


class TRTSeg:
    """SegFormer TRT 엔진 1개를 감싼 추론기. infer(bgr) → HxW int64 클래스맵."""
    kind = "trt"

    def __init__(self, domain):
        import tensorrt as trt
        import torch
        self._torch = torch

        meta = json.loads(labels_path(domain).read_text())
        self.domain = domain
        self.size = int(meta["input_size"])
        self.id2label = {int(k): v for k, v in meta["id2label"].items()}
        self.num_classes = int(meta["num_classes"])

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path(domain), "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"[{domain}] 엔진 역직렬화 실패")
        self.ctx = self.engine.create_execution_context()

        # IO 텐서 이름/모양/타입 파악
        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                self.in_name = nm
            else:
                self.out_name = nm
        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        in_dt = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.in_name))
        out_dt = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.out_name))

        dev = torch.device("cuda")
        self._in = torch.empty(in_shape, dtype=in_dt, device=dev).contiguous()
        self._out = torch.empty(out_shape, dtype=out_dt, device=dev).contiguous()
        self.ctx.set_tensor_address(self.in_name, self._in.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self._out.data_ptr())
        self._mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)

    def infer(self, bgr):
        torch = self._torch
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(np.ascontiguousarray(r)).cuda()      # [S,S,3] uint8
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)   # [1,3,S,S]
        t = (t - self._mean) / self._std
        self._in.copy_(t.to(self._in.dtype))
        stream = torch.cuda.current_stream().cuda_stream
        self.ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        small = self._out.argmax(1, keepdim=True).float()         # [1,1,S/4,S/4]
        up = torch.nn.functional.interpolate(small, size=(H, W), mode="nearest")
        return up[0, 0].to("cpu").numpy().astype("int64")


def get(domain):
    """도메인별 TRTSeg 캐시 로더. (trt_seg, id2label) 반환."""
    if domain not in _cache:
        _cache[domain] = TRTSeg(domain)
    seg = _cache[domain]
    return seg, seg.id2label
