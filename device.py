#!/usr/bin/env python3
"""추론 장치 결정 — 젯슨 GPU(온보드)와 노트북 CPU(시각화)를 한 곳에서 정한다.

이 프로젝트는 위성 온보드(젯슨) 저전력 실증이 목표지만, 시연·개발은 노트북에서도
되어야 한다. `.to('cuda')` 를 코드 곳곳에 흩뿌리면 "어디는 GPU, 어디는 CPU" 상태가
반드시 생기고, 특히 **조용히 CPU 로 도는 경로**는 느려진 걸 "젯슨이 원래 느리다"로
오해하게 만든다. 그래서 장치 결정은 이 파일 하나가 한다.

park-hyun-su/syntax_inference 의 segdemo/device.py 아이디어를 이 프로젝트
(전력계측·SegFormer·양쪽 구동)에 맞춰 간추렸다. 젯슨에서 조심할 4가지:

  1) **CUDA 는 비동기다.** model(x) 는 커널을 큐에 넣고 즉시 돌아온다. 시간/에너지를
     재기 전 sync() 를 부르지 않으면 추론시간이 실제의 1/100 로 찍혀 mJ/frame 이
     거짓이 된다. 이 프로젝트의 수치는 전부 실측이어야 하므로 측정 경로는 sync().
  2) **Orin 8GB 는 통합메모리다.** OS·브라우저가 같은 8GB 를 쓴다. OOM 이면 앱이
     죽는 대신 CPU 로 내려간다(oom_fallback).
  3) **aarch64 pip torch 는 CPU 빌드다.** --device cuda 인데 CUDA 가 없으면 조용히
     CPU 로 가지 않고 왜 없는지 말하며 죽는다(느린 것보다 안 도는 게 낫다).
  4) **fp16 은 Orin(compute 8.7) 텐서코어 이득.** CPU 에서는 fp16 이 오히려 느려 끈다.
"""
from __future__ import annotations

import os
import platform
import threading

MODES = ("auto", "cpu", "cuda")
ENV_VAR = "ORBITAL_DEVICE"

_lock = threading.Lock()
_mode = os.environ.get(ENV_VAR, "auto").strip().lower() or "auto"
if _mode not in MODES:
    _mode = "auto"
_resolved: str | None = None      # "cpu" | "cuda"
_half: bool = False
_note: str = ""


def set_mode(mode: str) -> None:
    """모델을 올리기 전에만 부를 것."""
    global _mode, _resolved, _half, _note
    mode = (mode or "auto").strip().lower()
    if mode not in MODES:
        raise ValueError(f"device 는 {list(MODES)} 중 하나여야 함: {mode!r}")
    with _lock:
        _mode, _resolved, _half, _note = mode, None, False, ""


def _is_jetson() -> bool:
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return b"jetson" in f.read().lower()
    except OSError:
        return False


def _no_cuda_reason() -> str:
    if platform.machine().lower() in ("aarch64", "arm64") and _is_jetson():
        return ("젯슨인데 CUDA 를 못 쓴다 — PyPI 의 aarch64 torch 는 CPU 빌드다. "
                "NVIDIA JetPack 용 휠이 필요하다 (run.sh 의 LD_LIBRARY_PATH 확인).")
    return "이 기계에 CUDA 장치가 없다 (CPU 로 돈다)"


def _detect() -> tuple[str, bool, str]:
    if _mode == "cpu":
        return "cpu", False, "사용자가 --device cpu 로 고정"
    try:
        import torch
    except Exception as e:                                   # noqa: BLE001
        if _mode == "cuda":
            raise RuntimeError(f"torch 를 못 불러와 CUDA 사용 불가: {e}") from e
        return "cpu", False, f"torch 없음 ({e})"
    if not torch.cuda.is_available():
        why = _no_cuda_reason()
        if _mode == "cuda":
            raise RuntimeError(f"--device cuda 인데 CUDA 를 못 쓴다. {why}")
        return "cpu", False, why
    try:
        major = torch.cuda.get_device_capability(0)[0]
    except Exception:                                        # noqa: BLE001
        major = 0
    half = major >= 6
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:                                        # noqa: BLE001
        name = "CUDA"
    return "cuda", half, f"{name} · fp16 {'켬' if half else '끔'}"


def _ensure() -> None:
    global _resolved, _half, _note
    if _resolved is not None:
        return
    with _lock:
        if _resolved is None:
            _resolved, _half, _note = _detect()


def mode() -> str:
    return _mode


def name() -> str:
    _ensure()
    return _resolved or "cpu"


def is_cuda() -> bool:
    return name() == "cuda"


def use_half() -> bool:
    _ensure()
    return bool(_half)


def torch_device():
    import torch
    return torch.device(name())


def note() -> str:
    _ensure()
    return _note


def to_model(model):
    """모델을 현재 장치·dtype 으로. 양방향(CUDA→CPU 도)이며 fp16 캐스팅까지 여기서."""
    if not is_cuda():
        return model.float().cpu().eval()
    model = model.to(torch_device())
    if use_half():
        model = model.half()
    return model.eval()


def to_inputs(pixel_values):
    """전처리 텐서를 장치·dtype 에 맞춘다. 실수만 half (정수 마스크는 그대로)."""
    import torch
    if not isinstance(pixel_values, torch.Tensor):
        return pixel_values
    if not is_cuda():
        return pixel_values
    t = pixel_values.to(torch_device())
    if use_half() and t.is_floating_point():
        t = t.half()
    return t


def sync() -> None:
    """CUDA 큐가 빌 때까지 대기. **시간/에너지 측정 직전·직후에 반드시.**"""
    if not is_cuda():
        return
    import torch
    try:
        torch.cuda.synchronize()
    except Exception:                                        # noqa: BLE001, S110
        pass


def empty_cache() -> None:
    if not is_cuda():
        return
    import torch
    try:
        torch.cuda.empty_cache()
    except Exception:                                        # noqa: BLE001, S110
        pass


def is_oom(exc: BaseException) -> bool:
    import torch
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ()) or ()):
        return True
    return "out of memory" in str(exc).lower()


def summary() -> str:
    """한 줄 요약 — 앱 기동·리포트 머리말에 그대로 쓴다."""
    if not is_cuda():
        return f"CPU  ({note()})"
    import torch
    try:
        gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:                                        # noqa: BLE001
        gb = 0.0
    prec = "fp16" if use_half() else "fp32"
    return f"CUDA · {prec} · {gb:.1f}GB{' 공유(젯슨)' if _is_jetson() else ''}  ({note()})"


def diagnose() -> str:
    lines = [f"기계      {platform.machine()} · {platform.system()}"]
    if _is_jetson():
        try:
            with open("/proc/device-tree/model", "rb") as f:
                lines.append(f"젯슨      {f.read().decode('utf-8', 'replace').strip(chr(0)).strip()}")
        except OSError:
            pass
    try:
        import torch
        lines += [f"torch     {torch.__version__}",
                  f"CUDA 빌드 {torch.version.cuda or '없음 (CPU 빌드)'}",
                  f"CUDA 가용 {torch.cuda.is_available()}"]
    except Exception as e:                                   # noqa: BLE001
        lines.append(f"torch     못 불러옴: {e}")
    lines += [f"고른 장치 {name()} · fp16 {use_half()}", f"이유      {note()}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(diagnose())
