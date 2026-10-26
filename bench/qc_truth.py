"""Truth for scoring QC flags on the Xenium ROIs (10x masks + nucleus centroids as reference).

  doublet flag on a SAM cell is TRUE if that cell contains >= 2 10x nucleus centroids
  missed  flag is TRUE if its blob centre lies in a 10x cell that no SAM cell matches (IoU 0.5) and that holds no SAM seed
  empty   flag is TRUE if the SAM cell contains no 10x nucleus centroid

Shared by bench/qc_check.py and bench/qc_muse_smoke.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def matched_ids(gt: np.ndarray, pred: np.ndarray, thr: float = 0.5) -> set[int]:
    """Set of gt ids matched one-to-one at IoU >= thr (same greedy rule as match_f1)."""
    gt_ids = np.unique(gt); gt_ids = gt_ids[gt_ids > 0]; pr_ids = np.unique(pred); pr_ids = pr_ids[pr_ids > 0]
    if not len(gt_ids) or not len(pr_ids):
        return set()
    g = np.searchsorted(gt_ids, gt) + 1; g[gt == 0] = 0
    p = np.searchsorted(pr_ids, pred) + 1; p[pred == 0] = 0
    ng, npr = len(gt_ids) + 1, len(pr_ids) + 1
    inter = np.bincount(g.ravel() * npr + p.ravel(), minlength=ng * npr).reshape(ng, npr)
    union = inter.sum(1, keepdims=True) + inter.sum(0, keepdims=True) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0)[1:, 1:]
    used_g = np.zeros(iou.shape[0], bool); used_p = np.zeros(iou.shape[1], bool); out = set()
    for flat in np.argsort(iou, axis=None)[::-1]:
        if iou.flat[flat] < thr:
            break
        i, j = divmod(int(flat), iou.shape[1])
        if not used_g[i] and not used_p[j]:
            used_g[i] = used_p[j] = True; out.add(int(gt_ids[i]))
    return out


def centroids_px(lab: np.ndarray) -> np.ndarray:
    ids = np.arange(1, lab.max() + 1)
    c = np.array(ndi.center_of_mass(lab > 0, lab, index=ids), dtype=float)
    c = c[np.isfinite(c).all(axis=1)]
    return np.c_[np.clip(c[:, 0].round(), 0, lab.shape[0] - 1), np.clip(c[:, 1].round(), 0, lab.shape[1] - 1)].astype(int)


def truth_sets(gt_c: np.ndarray, gt_n: np.ndarray, sam: np.ndarray, seeds: np.ndarray) -> dict:
    """true_dbl / true_empty = sets of SAM cell ids; truly_missed = set of 10x cell ids; plus n_10x, n_matched."""
    nc = centroids_px(gt_n)
    per_sam = np.bincount(sam[nc[:, 0], nc[:, 1]], minlength=sam.max() + 1)[1:]       # 10x nuclei per SAM cell
    area = np.bincount(sam.ravel(), minlength=sam.max() + 1)[1:]
    true_dbl = set(np.where(per_sam >= 2)[0] + 1)
    true_empty = set(np.where((per_sam == 0) & (area > 0))[0] + 1)
    matched = matched_ids(gt_c, sam)
    sc = centroids_px(seeds); seed_in = np.zeros(gt_c.max() + 1, bool); seed_in[gt_c[sc[:, 0], sc[:, 1]]] = True
    gt_ids = np.unique(gt_c); gt_ids = gt_ids[gt_ids > 0]
    truly_missed = {int(i) for i in gt_ids if i not in matched and not seed_in[i]}
    return dict(true_dbl=true_dbl, true_empty=true_empty, truly_missed=truly_missed, n_10x=int(len(gt_ids)), n_matched=len(matched))


def flag_truth(flags: dict[str, pd.DataFrame], truth: dict, gt_c: np.ndarray) -> dict[str, pd.Series]:
    """One boolean Series per kind (doublet, missed, empty), index-aligned with the flag table."""
    out = {}
    out["doublet"] = pd.Series([int(c) in truth["true_dbl"] for c in flags["doublet"].cell], index=flags["doublet"].index, dtype=bool)
    hit = [int(gt_c[int(round(r.cy)), int(round(r.cx))]) for r in flags["missed"].itertuples()]
    out["missed"] = pd.Series([h in truth["truly_missed"] for h in hit], index=flags["missed"].index, dtype=bool)
    out["missed_in_any_10x"] = pd.Series([h > 0 for h in hit], index=flags["missed"].index, dtype=bool)
    out["empty"] = pd.Series([int(c) in truth["true_empty"] for c in flags["empty"].cell], index=flags["empty"].index, dtype=bool)
    return out


def prf(flags_true, n_true_total: int) -> tuple[float, float]:
    flags_true = list(flags_true)
    p = float(np.mean(flags_true)) if len(flags_true) else float("nan")
    r = float(sum(flags_true) / n_true_total) if n_true_total else float("nan")
    return p, r
