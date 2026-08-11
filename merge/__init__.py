"""
merge — 형준(구간02, 온보드 분할)과 박현수(구간03, 전력)의 두 프로그램을 같은 자로
재고 항목별로 채택하기 위한 계약·계측·채택 도구.

박현수 「젯슨 현황과 병합 계획」에서 이름만 있고 아직 구현되지 않았던
`segdemo.jetson`(계측)·`segdemo.merge`(채택)를, orbital-perception 인프라
(powerlog·device)를 재사용해 정정본으로 구현한 것이다. 모델 무관 —
형준 프로그램은 파이썬이 아니어도 `--seg-cmd` 로 그대로 끼운다.

  python -m merge contract              계약(클래스표·밴드·규격)을 코드에서 출력
  python -m merge check out.npy --shape 2048,2048
  python -m merge idle                  유휴 전력(W) 측정
  python -m merge measure --seg-cmd "..." --in x.npy --measure-idle --out a.json
  python -m merge adopt a.json b.json --incumbent 우리
  python -m merge selftest              전 과정·수식 자체검증(젯슨 없이도 됨)
"""
from . import adopt, contract, meter        # noqa: F401

__all__ = ["contract", "meter", "adopt"]
