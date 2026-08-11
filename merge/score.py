#!/usr/bin/env python3
"""
merge.score — 정확도(mIoU) 실측. 형준 축(분할 품질)을 손입력 대신 잰다.

이 도구는 박현수 축(전력)을 idle 차감·열스로틀·표A 게이트로 엄밀히 재면서, 정작
형준의 기여 전부인 '분할 정확도'는 `--miou 0.68` 로 **손입력**받아 왔다. 저울의
반쪽이 눈대중이면 채택표를 믿을 수 없다 — 여기서 예측(pred)·정답(label) npy 를
받아 `contract` 의 10-클래스표로 per-class IoU·mIoU 를 **실측**한다.

  · 혼동행렬 수식은 orbital-perception `metrics.ConfusionMatrix` 를 **그대로 재사용**한다.
    계약을 한 곳에서만 두듯 지표도 한 곳에서만 둔다 — 복제하면 두 mIoU 가 갈린다.
  · 라벨의 무효화소(nodata)는 ignore_index(기본 255)와 -1 로 빼고, 라벨에 실제
    등장한 클래스만 평균(present)해 넓은 클래스가 점수를 지배하지 않게 한다.
  · 예측은 계약(uint8, 0~9)을 지켜야 하고, 라벨은 nodata 를 허용해 조금 관대하다.
  · 크기가 다르거나 계약을 어기면 조용히 넘기지 않고 멈춘다(fail-loud).
"""
from __future__ import annotations

import numpy as np

from . import contract

DEFAULT_IGNORE = 255          # 라벨 무효화소 관례값. -1 도 자동으로 무효 처리된다.
N_WORST = 5


class ScoreError(ValueError):
    """정확도를 잴 수 없는 상태(예측 계약 위반·크기 불일치·라벨 규격 밖)."""


def _confusion_matrix():
    """지표 수식은 한 곳에서만 — orbital-perception/metrics.py 를 재사용."""
    from metrics import ConfusionMatrix
    return ConfusionMatrix


def validate_label(arr: np.ndarray, *, path: str | None = None,
                   ignore_index: int = DEFAULT_IGNORE) -> np.ndarray:
    """정답 {label} 검사. (H,W) 2D, 값은 0~9 또는 무효(ignore_index/-1).
    예측보다 관대하다(nodata 허용). 통과하면 int64 배열로 돌려준다."""
    where = f"{path}: " if path else ""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ScoreError(f"{where}라벨은 (H,W) 2차원이어야 하는데 {a.ndim}차원 {a.shape} 이다.")
    if not np.issubdtype(a.dtype, np.integer):
        # float 라벨이 정수값이면 받아준다(흔한 실수). NaN/소수면 거부.
        if not np.all(np.isfinite(a)) or not np.all(a == np.round(a)):
            raise ScoreError(f"{where}라벨은 정수 클래스 id 여야 하는데 dtype 이 {a.dtype} 이다.")
        a = a.astype(np.int64)
    ids = set(np.unique(a).tolist())
    valid = set(range(contract.N_CLASSES)) | {ignore_index, -1}
    bad = sorted(ids - valid)
    if bad:
        raise ScoreError(
            f"{where}라벨에 규격 밖 값 {bad} 이 있다(허용 0~{contract.MAX_ID}, "
            f"무효 {ignore_index}/-1). 번호표는 `python -m merge contract`.")
    return a.astype(np.int64)


def score_arrays(pred: np.ndarray, label: np.ndarray, *,
                 ignore_index: int = DEFAULT_IGNORE) -> dict:
    """예측·정답으로 per-class IoU·mIoU 를 낸다.

    pred 는 계약(uint8 0~9)을 지켜야 하고, label 은 무효화소를 허용한다. 크기가
    다르면 멈춘다. 반환은 JSON 직렬화 가능한 dict — `accuracy.miou` 로 adopt 가 읽는다.
    """
    ConfusionMatrix = _confusion_matrix()
    try:
        pred = contract.validate_output(np.asarray(pred))
    except contract.ContractError as e:
        raise ScoreError(f"예측이 계약(uint8, 0~9)을 어겼다 — 정확도 측정 불가: {e}")
    label = validate_label(label, ignore_index=ignore_index)
    if pred.shape != label.shape:
        raise ScoreError(
            f"예측 {pred.shape} 와 라벨 {label.shape} 크기가 다르다 → 정확도 측정 불가. "
            f"둘 다 같은 장면의 (H,W) 여야 한다.")

    # ConfusionMatrix.update 의 keep 이 label==ignore, label<0, 범위 밖을 모두 배제한다.
    cm = ConfusionMatrix(contract.N_CLASSES, ignore_index=ignore_index)
    cm.update(pred, label)

    iou = cm.per_class_iou()
    present = cm.present()
    per: dict[str, float] = {}
    for cid in np.where(present)[0]:
        v = iou[cid]
        if np.isnan(v):
            continue
        name = contract.BY_ID[int(cid)].ko if int(cid) in contract.BY_ID else str(int(cid))
        per[name] = round(float(v), 4)

    miou = cm.miou()
    pacc = cm.pixel_acc()
    return {
        "miou": None if np.isnan(miou) else round(float(miou), 4),
        "pixel_acc": None if np.isnan(pacc) else round(float(pacc), 4),
        "per_class_iou": per,
        "worst_classes": sorted(per.items(), key=lambda kv: kv[1])[:N_WORST],
        "n_classes_present": int(present.sum()),
        "n_valid_px": int(cm.mat.sum()),
        "ignore_index": ignore_index,
    }


def score_files(pred_path: str, label_path: str, *,
                ignore_index: int = DEFAULT_IGNORE) -> dict:
    """예측·정답 npy 파일 경로로 점수를 낸다. 계약/크기 문제는 ScoreError 로 올린다."""
    pred = np.load(pred_path)
    label = np.load(label_path)
    return score_arrays(pred, label, ignore_index=ignore_index)
