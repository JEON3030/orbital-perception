"""merge e2e — measure 경로를 dummy_seg.py 서브프로세스로 실검증한다.

순수 로직만 보는 test_merge.py 와 달리, 여기선 실제 외부 프로그램을 --seg-cmd 로
끼워 돌리며 계약→실행→출력검증→스로틀→에너지 배관이 실제로 이어지는지 본다.
전력 계측은 PowerMeter 가 tegrastats 없으면 sysfs 로 폴백(예외 없음)하므로 젯슨이
아니어도 measure() 자체는 완주한다 — 값이 아니라 '배관이 이어지는가'를 검증한다.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from merge import contract, meter

DUMMY = Path(__file__).parent / "dummy_seg.py"


def _make_scene(tmp_path, h=16, w=16):
    """유효한 (H,W,6) float32 반사율[0,1] 장면을 만들어 경로를 돌려준다."""
    x = np.full((h, w, contract.N_BANDS), 0.2, np.float32)
    p = tmp_path / "scene.npy"
    np.save(p, x)
    return str(p)


def _cmd(*extra):
    """dummy_seg 를 이 파이썬으로 부르는 --seg-cmd 문자열({in}/{out} 자리표시)."""
    return f"{sys.executable} {DUMMY} {' '.join(extra)} --in {{in}} --out {{out}}".strip()


# ── run_seg_cmd + 계약 배관 (전력 불필요) ────────────────────────────────
def test_dummy_seg_output_obeys_contract(tmp_path):
    inp = _make_scene(tmp_path)
    out = str(tmp_path / "out.npy")
    dt = meter.run_seg_cmd(_cmd(), inp, out)
    assert dt >= 0.0
    y = contract.load_output(out, shape=(16, 16))     # 계약을 통과해야 한다
    assert y.dtype == np.uint8 and 0 <= int(y.min()) and int(y.max()) <= 9


def test_dummy_seg_bad_output_caught_by_contract(tmp_path):
    inp = _make_scene(tmp_path)
    out = str(tmp_path / "bad.npy")
    meter.run_seg_cmd(_cmd("--bad"), inp, out)         # 일부러 int64 로 낸다
    with pytest.raises(contract.ContractError):
        contract.load_output(out, shape=(16, 16))      # 검증기가 잡아야 한다


def test_run_seg_cmd_nonzero_exit_is_loud(tmp_path):
    inp = _make_scene(tmp_path)
    out = str(tmp_path / "o.npy")
    with pytest.raises(RuntimeError):                  # 조용히 넘기지 않는다
        meter.run_seg_cmd(f'{sys.executable} -c "import sys; sys.exit(3)"', inp, out)


def test_run_seg_cmd_timeout_is_loud(tmp_path):
    inp = _make_scene(tmp_path)
    out = str(tmp_path / "o.npy")
    with pytest.raises(RuntimeError):
        meter.run_seg_cmd(f'{sys.executable} -c "import time; time.sleep(5)"',
                          inp, out, timeout=0.5)


def test_throttle_flag_detects_slowdown():
    warn, ratio = meter._throttle_flag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    assert warn and ratio >= meter.THROTTLE_RATIO
    warn2, _ = meter._throttle_flag([1.0, 1.0, 1.0])   # 안정적 → 경고 없음
    assert not warn2


# ── measure() 전체 배관 (PowerMeter 폴백으로 젯슨 없이도 완주) ─────────────
def test_measure_full_pipeline(tmp_path):
    inp = _make_scene(tmp_path)
    res = meter.measure(
        name="dummy", seg_cmd=_cmd(), input_npy=inp, reps=3,
        idle_watt=None, min_free_mb=0,                 # 메모리 게이트는 여기서 검증 대상 아님
    )
    assert res.contract_ok is True
    assert res.n_frames == 3 and len(res.per_rep_sec) == 3
    assert isinstance(res.energy, dict) and "mJ_per_frame" in res.energy
    # idle 미지정 → dynamic 없음 + 그 사실을 note 로 남긴다(속도=에너지 판단 불가 안내)
    assert any("idle" in n for n in res.notes)


def test_measure_refuses_low_memory(tmp_path):
    inp = _make_scene(tmp_path)
    with pytest.raises(RuntimeError):                  # 표 A 게이트: 성능이 아니라 '운'
        meter.measure(name="x", seg_cmd=_cmd(), input_npy=inp, reps=3,
                      idle_watt=None, min_free_mb=10**9)


def test_measure_rejects_bad_input_contract(tmp_path):
    # (H,W,3) → 밴드 수 위반 → 측정 진입 전에 멈춘다
    bad = tmp_path / "bad_in.npy"
    np.save(bad, np.zeros((8, 8, 3), np.float32))
    with pytest.raises(contract.ContractError):
        meter.measure(name="x", seg_cmd=_cmd(), input_npy=str(bad), reps=3,
                      idle_watt=None, min_free_mb=0)
