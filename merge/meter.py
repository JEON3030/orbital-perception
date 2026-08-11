#!/usr/bin/env python3
"""
merge.meter — 같은 자로 재는 계측기 (박현수 `segdemo.jetson` 을 정정본으로 구현).

박현수 문서의 함정을 그대로 방어하고, 앞선 리뷰에서 잡은 두 정정을 코드로 박았다.

  1. 에너지 회계 정정 — total 과 dynamic(idle 차감) mJ/frame 을 **항상 같이** 낸다.
     "2배 빨라도 전기 7%" 는 board-on-window 레짐에서만 참이다. 촬영→처리→대기
     듀티사이클에서는 속도가 곧 에너지다. 두 수치를 나란히 내야 그 판단을 할 수 있다.
     (powerlog 의 idle_watt 차감을 그대로 사용.)
  2. GPU fail-loud — GPU 를 요구했는데 없으면 조용히 CPU 로 안 내려가고 멈춘다.
  3. 표 A(고정 조건) 캡처 — 보드·전력모드·빈 메모리·반복 횟수를 결과에 같이 남긴다.
     빈 메모리가 부족하면 "성능이 아니라 운을 재는 것"이라 측정을 거부한다.
  4. 열 스로틀 경고 — 회차마다 느려지면(36→64→81초 함정) 경고 플래그를 세운다.

형준 프로그램은 파이썬이 아니어도 된다. `--seg-cmd` 로 외부 실행 명령을 그대로 끼운다.
계측기는 보드 총전력만 진실로 잰다 — 외부 프로그램이 실제로 GPU 를 썼는지는 그
프로그램이 스스로 밝혀야 한다(fail-loud 는 그쪽 몫). 이 경계를 결과에 명시한다.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import contract

# ── 표 A : 비교할 때 반드시 고정할 것 ────────────────────────────────────
DEFAULT_MIN_FREE_MB = 2000        # 시작 시 빈 메모리 최소치 (박현수: 2~3GB)
DEFAULT_REPEAT = 3                # 최소 반복 (한 번만 재면 결론이 뒤집힌다)
THROTTLE_RATIO = 1.3             # 뒤 1/3 이 앞 1/3 보다 이 배 이상 느리면 열 스로틀 의심


@dataclass
class TableA:
    """비교 무효를 판정하는 고정 조건. adopt 가 두 결과의 이 값을 대조한다."""
    board: str
    power_mode: str
    scene_px: int | None
    mem_free_mb: int
    mem_available_mb: int
    repeat: int
    device_env: str            # 이 계측 환경의 torch 장치 (참고용)
    tool: str = "merge.meter"
    when: str = ""


def _read_board() -> str:
    try:
        s = Path("/proc/device-tree/model").read_bytes().decode(errors="ignore").strip("\x00").strip()
        return s or "unknown"
    except OSError:
        return "unknown"


def _read_power_mode() -> str:
    """nvpmodel -q 로 전력 모드. 젯슨이 아니면 'n/a'."""
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return "n/a"
    m = re.search(r"NV Power Mode:\s*(\S+)", out)
    if m:
        return m.group(1)
    m = re.search(r"pmode\D*(\d+)", out, re.I)
    return f"mode{m.group(1)}" if m else "unknown"


def _read_mem_mb() -> tuple[int, int]:
    """(MemFree, MemAvailable) in MB. available 은 낙관적일 수 있음(주의는 CLI 가 출력)."""
    free = avail = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemFree:"):
                free = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) // 1024
    except OSError:
        pass
    return free, avail


def capture_table_a(*, scene_px: int | None, repeat: int) -> TableA:
    free, avail = _read_mem_mb()
    try:
        import device  # orbital-perception 의 fail-loud 장치 선택기
        dev = device.name()
    except Exception as e:                       # noqa: BLE001 — torch 없는 환경도 계측은 가능
        dev = f"unknown ({type(e).__name__})"
    return TableA(
        board=_read_board(), power_mode=_read_power_mode(), scene_px=scene_px,
        mem_free_mb=free, mem_available_mb=avail, repeat=repeat, device_env=dev,
        when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ── GPU fail-loud ────────────────────────────────────────────────────
def require_gpu_or_die() -> None:
    """GPU 를 요구했는데 이 환경이 CUDA 를 못 쓰면 멈춘다. 조용한 CPU 강등 금지."""
    try:
        import device
    except Exception as e:                       # noqa: BLE001
        raise RuntimeError(f"--require-gpu: torch/device 를 못 불렀다 ({e}). GPU 확인 불가라 멈춘다.")
    if not device.is_cuda():
        raise RuntimeError(
            "--require-gpu 인데 이 환경은 CUDA 를 못 쓴다 → 멈춘다(조용한 CPU 강등 금지).\n"
            f"  이유: {getattr(device, 'note', lambda: '?')()}\n"
            "  이 보드에서 GPU 를 켜는 두 길: (a) 매칭 torch 를 격리 설치(PYTHONPATH),\n"
            "  (b) TensorRT 엔진 경로(seg_trt). 외부 --seg-cmd 프로그램은 그쪽이 스스로 fail-loud 해야 한다.")


# ── seg 실행 (외부 명령) ──────────────────────────────────────────────
def run_seg_cmd(seg_cmd: str, in_path: str, out_path: str, *, timeout: float = 600) -> float:
    """{in}/{out} 자리를 채워 외부 프로그램을 돌린다. 걸린 벽시계 시간(초)을 돌려준다.
    실패(0 아님 종료·타임아웃)는 조용히 넘기지 않고 예외로 올린다(fail-loud)."""
    cmd = seg_cmd.replace("{in}", in_path).replace("{out}", out_path)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"seg-cmd 가 {timeout}s 안에 안 끝났다: {cmd}")
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(f"seg-cmd 종료코드 {p.returncode}: {cmd}\n  " + "\n  ".join(tail))
    return dt


def _throttle_flag(times: list[float]) -> tuple[bool, float]:
    """회차 시간 목록에서 뒤가 앞보다 느려졌는지. (경고여부, 배율)."""
    if len(times) < 3:
        return False, 1.0
    k = max(1, len(times) // 3)
    head = float(np.median(times[:k]))
    tail = float(np.median(times[-k:]))
    ratio = tail / head if head > 0 else 1.0
    return ratio >= THROTTLE_RATIO, round(ratio, 3)


@dataclass
class MeterResult:
    name: str
    table_a: dict
    seg_cmd: str
    contract_ok: bool
    n_frames: int
    per_rep_sec: list[float]
    throttle_warn: bool
    throttle_ratio: float
    energy: dict = field(default_factory=dict)   # powerlog.report(...) 그대로
    accuracy: dict = field(default_factory=dict) # 정확도는 따로 넣는다(측정 주체 다름)
    notes: list[str] = field(default_factory=list)


def measure(*, name: str, seg_cmd: str, input_npy: str, reps: int = DEFAULT_REPEAT,
            idle_watt: float | None = None, require_gpu: bool = False,
            min_free_mb: int = DEFAULT_MIN_FREE_MB, out_json: str | None = None,
            interval_ms: int = 100) -> MeterResult:
    """seg_cmd 를 reps 회 돌리며 total/dynamic mJ/frame 을 잰다. 표 A 를 같이 남긴다."""
    notes: list[str] = []

    # 입력 계약 검사 (틀리면 여기서 멈춘다)
    x = contract.load_input(input_npy)
    H, W = x.shape[:2]
    scene_px = max(H, W)

    # 표 A + 게이트
    if require_gpu:
        require_gpu_or_die()
    ta = capture_table_a(scene_px=scene_px, repeat=reps)
    if ta.mem_free_mb < min_free_mb:
        raise RuntimeError(
            f"빈 메모리 {ta.mem_free_mb}MB < 필요 {min_free_mb}MB → 측정 거부.\n"
            f"  꽉 찬 상태면 성능이 아니라 '운'을 잰다(박현수 표 A). available({ta.mem_available_mb}MB)는\n"
            f"  회수 가능한 캐시를 포함해 낙관적이다 — 대형 CUDA 할당엔 못 믿는다. 비우고 다시.")
    if reps < DEFAULT_REPEAT:
        notes.append(f"반복 {reps}회 < 권장 {DEFAULT_REPEAT}회 — 결론이 뒤집힐 수 있다.")

    out_npy = str(Path(input_npy).with_suffix(".seg_out.npy"))

    # 워밍업 1회 (첫 호출은 모델 로딩으로 느리다 — 측정에서 뺀다)
    run_seg_cmd(seg_cmd, input_npy, out_npy)
    contract_ok = True
    try:
        contract.load_output(out_npy, shape=(H, W))
    except contract.ContractError as e:
        contract_ok = False
        notes.append(f"출력 계약 위반: {e}")

    # 본 측정: PowerMeter 가 전 회차를 감싼다(보드 총전력), per-rep 은 벽시계
    from powerlog import PowerMeter
    per_rep: list[float] = []
    pm = PowerMeter(interval_ms=interval_ms)
    pm.start()
    for _ in range(reps):
        per_rep.append(run_seg_cmd(seg_cmd, input_npy, out_npy))
    pm.stop()

    energy = pm.report(n_frames=reps, idle_watt=idle_watt)
    if idle_watt is None:
        notes.append("idle_watt 미지정 → dynamic(순수) mJ/frame 없음. `measure idle` 로 먼저 재면 "
                     "'속도=에너지'가 성립하는지 판단할 수 있다.")
    warn, ratio = _throttle_flag(per_rep)
    if warn:
        notes.append(f"열 스로틀 의심: 뒤 회차가 앞의 {ratio}배로 느려짐. fps/에너지 결론을 믿지 말 것.")

    res = MeterResult(
        name=name, table_a=asdict(ta), seg_cmd=seg_cmd, contract_ok=contract_ok,
        n_frames=reps, per_rep_sec=[round(t, 4) for t in per_rep],
        throttle_warn=warn, throttle_ratio=ratio, energy=energy, notes=notes,
    )
    if out_json:
        Path(out_json).write_text(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    return res


def measure_idle(seconds: float = 3.0) -> float:
    """유휴 전력(W). powerlog 것을 그대로 쓰되 없으면 0 을 돌려주고 알린다."""
    try:
        from powerlog import measure_idle as _mi
        return float(_mi(seconds))
    except Exception as e:                       # noqa: BLE001
        print(f"[warn] idle 측정 실패({e}) → 0 W 로 둔다(젯슨 아님?). dynamic 수치는 무의미해진다.")
        return 0.0
