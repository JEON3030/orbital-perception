#!/usr/bin/env python3
"""테스트용 더미 분할기 — 계약대로 (H,W,6)f32 를 읽어 (H,W)u8 클래스맵을 쓴다.
실제 모델이 아니라 계측/계약 파이프라인을 검증하기 위한 자리표시자다.
--bad 를 주면 일부러 계약을 위반(int64)해 검증기가 잡는지 확인한다."""
import argparse

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", required=True)
p.add_argument("--bad", action="store_true")
p.add_argument("--work", type=int, default=200, help="가짜 계산량(측정거리용)")
a = p.parse_args()

x = np.load(a.inp)                       # (H,W,6) f32
h, w = x.shape[:2]
# 밴드 몇 개로 아주 단순한 지표를 만들어 클래스로 나눈다(내용은 중요치 않음).
s = x.sum(axis=2)
for _ in range(a.work):                  # 시간이 걸리도록 약간의 계산
    s = s + np.sin(s) * 0.0
seg = (np.abs(s) * 3).astype(np.int64) % 10
np.save(a.out, seg if a.bad else seg.astype(np.uint8))
