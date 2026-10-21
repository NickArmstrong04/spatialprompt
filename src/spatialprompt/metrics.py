"""Instance-segmentation metrics shared by the benchmarks."""
from __future__ import annotations

import numpy as np


def match_f1(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5) -> dict:
    """Matched-instance precision/recall/F1 at an IoU threshold (greedy one-to-one by max IoU).

    `gt` and `pred` are label images (0 = background, any integer ids)."""
    gt = np.asarray(gt).astype(np.int64); pred = np.asarray(pred).astype(np.int64)
    gt_ids = np.unique(gt); gt_ids = gt_ids[gt_ids > 0]
    pr_ids = np.unique(pred); pr_ids = pr_ids[pr_ids > 0]
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return dict(tp=0, fp=len(pr_ids), fn=len(gt_ids), precision=0.0, recall=0.0, f1=0.0)
    gt_d = np.searchsorted(gt_ids, gt, side="left") + 1; gt_d[gt == 0] = 0
    pr_d = np.searchsorted(pr_ids, pred, side="left") + 1; pr_d[pred == 0] = 0
    ng, npred = len(gt_ids) + 1, len(pr_ids) + 1
    inter = np.bincount(gt_d.ravel() * npred + pr_d.ravel(), minlength=ng * npred).reshape(ng, npred)
    area_g = inter.sum(1, keepdims=True); area_p = inter.sum(0, keepdims=True)
    union = area_g + area_p - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)[1:, 1:]
    tp = 0
    used_g, used_p = np.zeros(iou.shape[0], bool), np.zeros(iou.shape[1], bool)
    order = np.argsort(iou, axis=None)[::-1]
    for flat in order:
        v = iou.flat[flat]
        if v < iou_thr:
            break
        i, j = divmod(int(flat), iou.shape[1])
        if not used_g[i] and not used_p[j]:
            used_g[i] = used_p[j] = True; tp += 1
    fp, fn = int(len(pr_ids) - tp), int(len(gt_ids) - tp)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9))
