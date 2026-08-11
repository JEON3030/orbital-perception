#!/usr/bin/env python3
"""
merge.adopt — 항목별 채택표 (박현수 표 B) 를 코드로. 결과를 보기 전에 규칙을 박는다.

박현수 병합 규칙을 그대로, 그리고 앞선 리뷰의 두 정정을 지켜 구현한다.

  · 누가 만들었는지로 정하지 않는다 — 같은 조건에서 재고 이긴 항목을 채택.
  · 5% 안쪽이면 동률 → 먼저 있던 쪽(incumbent) 유지. 바꾸는 비용이 이득보다 크다.
  · 안 잰 항목은 0 으로 채우지 않고 **"판정 불가"** 로 남긴다(0 이면 한쪽이 그냥 이긴다).
  · 표 A(고정 조건)가 다르면 표를 내기 전에 **경고하고 멈춘다**. 조건이 다른 비교는 무효다.
  · 프로그램을 통째로 고르지 않는다 — 항목마다 따로 채택("정확도는 A, 화면은 B" 가능).
  · 진 항목도 버리지 않는다 — baseline 으로 남겨 "왜 골랐는가"를 증명한다(요약에 명시).
"""
from __future__ import annotations

from dataclasses import dataclass

TIE_TOL = 0.05        # 상대차 5% 안쪽이면 동률


class InvalidComparison(ValueError):
    """표 A 조건이 달라 비교 자체가 무효."""


# ── 표 A 대조 : 이게 다르면 비교 무효 ────────────────────────────────────
def check_table_a(a: dict, b: dict) -> list[str]:
    """두 결과의 표 A 를 대조. 치명적 불일치가 있으면 예외, 경고는 리스트로 돌려준다."""
    ta, tb = a.get("table_a", {}), b.get("table_a", {})
    fatal, warn = [], []

    def board_key(s: str) -> str:      # "NVIDIA Orin Nano ..." → 대략 같은 보드류로 본다
        return (s or "").lower().split("developer")[0].strip()

    if board_key(ta.get("board", "")) != board_key(tb.get("board", "")):
        fatal.append(f"보드가 다르다: {ta.get('board')!r} vs {tb.get('board')!r}")
    if (ta.get("power_mode") or "?") != (tb.get("power_mode") or "?"):
        fatal.append(f"전력 모드가 다르다: {ta.get('power_mode')} vs {tb.get('power_mode')} — 느린 쪽이 억울해진다")
    if ta.get("scene_px") != tb.get("scene_px"):
        fatal.append(f"입력 크기가 다르다: {ta.get('scene_px')} vs {tb.get('scene_px')} — 어려운 사진 받은 쪽이 진다")
    for tag, t in (("A", ta), ("B", tb)):
        if (t.get("repeat") or 0) < 3:
            warn.append(f"{tag}: 반복 {t.get('repeat')}회 < 3 — 한 번만 재면 결론이 뒤집힌다")
    if a.get("throttle_warn"):
        warn.append("A: 열 스로틀 경고가 켜져 있다 — 속도/에너지 결론을 믿지 말 것")
    if b.get("throttle_warn"):
        warn.append("B: 열 스로틀 경고가 켜져 있다 — 속도/에너지 결론을 믿지 말 것")

    if fatal:
        raise InvalidComparison("표 A 불일치로 이 비교는 무효다:\n  - " + "\n  - ".join(fatal))
    return warn


# ── 항목 하나의 승패 ──────────────────────────────────────────────────
UNDECIDED = "판정 불가"
TIE = "동률"


def _rel_cmp(va: float | None, vb: float | None, higher_better: bool,
             tol: float = TIE_TOL) -> str:
    """A/B/동률/판정 불가 를 돌려준다. 한쪽이라도 None 이면 판정 불가."""
    if va is None or vb is None:
        return UNDECIDED
    hi = max(abs(va), abs(vb))
    if hi == 0:
        return TIE
    if abs(va - vb) / hi < tol:
        return TIE
    a_wins = (va > vb) if higher_better else (va < vb)
    return "A" if a_wins else "B"


def _median(xs) -> float | None:
    xs = [x for x in (xs or []) if x is not None]
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


@dataclass
class Item:
    key: str
    label: str
    raw_verdict: str          # A / B / 동률 / 판정 불가
    winner: str               # 실제 채택된 이름(동률이면 incumbent), 또는 판정 불가
    va: object
    vb: object
    who: str                  # 주로 누구 몫 (표 B)


def _val(res: dict, path: str):
    """'energy.dynamic_mJ_per_frame' 같은 점 경로로 값을 꺼낸다. 없으면 None."""
    cur = res
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# (key, 라벨, 값경로 or 특수, higher_better, 누구몫)
_SPEC = [
    ("accuracy", "정확도(mIoU)",   "accuracy.miou",                 True,  "전형준"),
    ("speed",    "속도(장당 초)",   "@speed",                        False, "전형준"),
    ("energy",   "전기(mJ/frame)",  "@energy",                       False, "박현수"),
    ("memory",   "메모리(MB)",      "accuracy.peak_mem_mb",          False, "박현수"),
    ("stability","안정성",          "@stability",                    True,  "공통"),
    ("display",  "화면(fps)",       "accuracy.display_fps",          True,  "전형준"),
    ("plug",     "붙이기 쉬움",      "@plug",                         True,  "공통"),
]


def _extract(res: dict, key: str, path: str):
    if path == "@speed":
        return _median(res.get("per_rep_sec"))
    if path == "@energy":
        # dynamic(순수) 이 있으면 그것으로, 없으면 total 로. 둘 다 없으면 None.
        e = res.get("energy", {})
        return e.get("dynamic_mJ_per_frame") or e.get("mJ_per_frame")
    if path == "@stability":
        # 안 죽고 계약 지키고 스로틀 없으면 1, 아니면 감점. 측정 자체가 있으면 판정 가능.
        if not res.get("energy"):
            return None
        score = 1.0
        if not res.get("contract_ok", True):
            score -= 0.5
        if res.get("throttle_warn"):
            score -= 0.3
        return round(score, 3)
    if path == "@plug":
        return 1.0 if res.get("contract_ok") else 0.0
    return _val(res, path)


def adopt(a: dict, b: dict, *, name_a: str = "A", name_b: str = "B",
          incumbent: str | None = None) -> dict:
    """두 결과로 표 B 를 만든다. incumbent(먼저 있던 쪽 이름)는 동률 시 유지된다."""
    warnings = check_table_a(a, b)          # 무효면 여기서 예외

    inc = incumbent or name_a
    if inc not in (name_a, name_b):
        raise ValueError(f"incumbent 는 {name_a!r} 또는 {name_b!r} 여야 한다(받은 값 {inc!r}).")

    items: list[Item] = []
    for key, label, path, higher, who in _SPEC:
        va, vb = _extract(a, key, path), _extract(b, key, path)
        verdict = _rel_cmp(_num(va), _num(vb), higher)
        if verdict == "A":
            winner = name_a
        elif verdict == "B":
            winner = name_b
        elif verdict == TIE:
            winner = inc                    # 5% 동률 → 기존 유지
        else:
            winner = UNDECIDED
        items.append(Item(key, label, verdict, winner, va, vb, who))

    adopted = {it.key: it.winner for it in items if it.winner != UNDECIDED}
    undecided = [it.label for it in items if it.winner == UNDECIDED]
    return {
        "name_a": name_a, "name_b": name_b, "incumbent": inc,
        "tie_tol": TIE_TOL, "warnings": warnings,
        "items": [it.__dict__ for it in items],
        "adopted": adopted, "undecided": undecided,
        "baseline_note": (f"진 항목의 프로그램도 버리지 말 것 — baseline 으로 남겨야 "
                          f"'왜 골랐는가'를 발표·논문에서 증명할 수 있다."),
    }


def _num(v):
    """비교용 숫자만 통과. bool/None/문자는 float 로 바꿀 수 있으면 바꾼다."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
