"""Score the QC detectors and the repair loop on the Xenium ROIs (SAM masks vs 10x).

Truth used for scoring (10x nucleus centroids + matching at IoU 0.5):
  doublet flag on a SAM cell is TRUE if that cell contains >= 2 10x nucleus centroids
  missed  flag is TRUE if its blob centre lies in a 10x cell that no SAM cell matches and that holds no SAM seed
  empty   flag is TRUE if the SAM cell contains no 10x nucleus centroid
Recall of each detector = flagged true cases / all such cases.

Then the agent loop (RulesPlanner) on SAM masks: whole-cell F1 vs 10x before / after, with
add-only and split-only ablations. Writes results/qc_check.csv, data/roi/<roi>/cells_sp_qc.tif,
qc_log.json, overlay_sp_qc.png (roi01: green = added cells, orange = split cells).

Usage: <spatial python> -X utf8 bench/qc_check.py [--only roi01] [--rounds 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from spatialprompt import Segmenter, make_composite, match_f1  # noqa: E402
from spatialprompt.qc import QCAgent, RulesPlanner, dapi_foreground, detect_all  # noqa: E402

TAG = "dapi_boundary_cell_t384_c0.3_d2"


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


def prf(flags_true: list[bool], n_true_total: int) -> tuple[float, float]:
    p = float(np.mean(flags_true)) if len(flags_true) else float("nan")
    r = float(sum(flags_true) / n_true_total) if n_true_total else float("nan")
    return p, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args()
    seg = Segmenter()
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / f"cells_sp_{TAG}.tif").exists())
    if a.only:
        rois = [p for p in rois if p.name in a.only]
    rows = []
    for roi in rois:
        t0 = time.time()
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        img = tifffile.imread(roi / "image.tif"); dapi = img[0]; rgb = make_composite(img, "dapi_boundary")
        gt_c = tifffile.imread(roi / "cells_10x.tif").astype(np.int64); gt_n = tifffile.imread(roi / "nuclei_10x.tif").astype(np.int64)
        sam = tifffile.imread(roi / f"cells_sp_{TAG}.tif").astype(np.int64)
        seeds = tifffile.imread(roi / f"nuclei_sp_{TAG}.tif").astype(np.int64)
        fg = dapi_foreground(dapi, px); fg_low = dapi_foreground(dapi, px, otsu_scale=0.45)
        # truth
        nc = centroids_px(gt_n)
        per_sam = np.bincount(sam[nc[:, 0], nc[:, 1]], minlength=sam.max() + 1)[1:]       # 10x nuclei per SAM cell
        true_dbl = set(np.where(per_sam >= 2)[0] + 1); true_empty = set(np.where((per_sam == 0) & (np.bincount(sam.ravel(), minlength=sam.max() + 1)[1:] > 0))[0] + 1)
        matched = matched_ids(gt_c, sam)
        sc = centroids_px(seeds); seed_in = np.zeros(gt_c.max() + 1, bool); seed_in[gt_c[sc[:, 0], sc[:, 1]]] = True
        gt_ids = np.unique(gt_c); gt_ids = gt_ids[gt_ids > 0]
        truly_missed = {int(i) for i in gt_ids if i not in matched and not seed_in[i]}
        # detectors
        fl = detect_all(sam, seeds, fg, px, fg_low=fg_low)
        d_true = [int(c) in true_dbl for c in fl["doublet"].cell]
        m_hit = [int(gt_c[int(round(r.cy)), int(round(r.cx))]) for r in fl["missed"].itertuples()]
        m_true = [h in truly_missed for h in m_hit]
        e_true = [int(c) in true_empty for c in fl["empty"].cell]
        dp, dr = prf(d_true, len(true_dbl)); mp, mr = prf(m_true, len(truly_missed)); ep, er = prf(e_true, len(true_empty))
        row = dict(roi=roi.name, n_sam=int(len(np.unique(sam)) - 1), n_10x=int(len(gt_ids)), n_unmatched_10x=int(len(gt_ids) - len(matched)),
                   true_doublets=len(true_dbl), true_missed=len(truly_missed), true_empty=len(true_empty),
                   flag_doublet=len(fl["doublet"]), flag_missed=len(fl["missed"]), flag_empty=len(fl["empty"]), flag_split=len(fl["split"]),
                   doublet_p=dp, doublet_r=dr, missed_p=mp, missed_r=mr, empty_p=ep, empty_r=er,
                   missed_in_any_10x=float(np.mean([h > 0 for h in m_hit])) if m_hit else float("nan"))
        # repairs
        before = match_f1(gt_c, sam)
        variants = {"full": RulesPlanner(), "add_only": RulesPlanner(do_split=False), "split_only": RulesPlanner(do_add=False)}
        logs = {}
        for name, planner in variants.items():
            cells, log = QCAgent(seg, planner).run(rgb, sam, seeds, dapi, px, rounds=a.rounds if name == "full" else 1, fg=fg)
            m = match_f1(gt_c, cells.astype(np.int64))
            row.update({f"f1_{name}": m["f1"], f"p_{name}": m["precision"], f"r_{name}": m["recall"], f"n_{name}": log["n_after"]})
            logs[name] = (cells, log)
        cells_full, log_full = logs["full"]
        acts = log_full["rounds"][0]["actions"]
        row.update(f1_before=before["f1"], p_before=before["precision"], r_before=before["recall"],
                   **{f"{k}_{m}": v[m] for k, v in acts.items() for m in ("planned", "applied")}, qc_sec=log_full["sec"], sec=time.time() - t0)
        rows.append(row)
        print(f"{roi.name}: flags dbl {len(fl['doublet'])} (P {dp:.2f} R {dr:.2f} of {len(true_dbl)}), missed {len(fl['missed'])} (P {mp:.2f} R {mr:.2f} of {len(truly_missed)}), "
              f"empty {len(fl['empty'])} (P {ep:.2f} R {er:.2f} of {len(true_empty)}), split {len(fl['split'])} | "
              f"F1 {before['f1']:.3f} -> full {row['f1_full']:.3f} (P {row['p_full']:.2f} R {row['r_full']:.2f}) | add-only {row['f1_add_only']:.3f} | split-only {row['f1_split_only']:.3f} "
              f"| {acts} [{log_full['sec']:.0f}s]", flush=True)
        tifffile.imwrite(roi / "cells_sp_qc.tif", cells_full)
        (roi / "qc_log.json").write_text(json.dumps(log_full, indent=2, default=float))
        if roi.name == "roi01":
            base = np.array(Image.open(roi / "composite.png")); y0 = x0 = (base.shape[0] - 1024) // 2
            sub = base[y0 : y0 + 1024, x0 : x0 + 1024].copy()
            s_new = cells_full[y0 : y0 + 1024, x0 : x0 + 1024].astype(np.int64)
            sub[find_boundaries(gt_c[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [0, 255, 255]
            sub[find_boundaries(s_new, mode="inner")] = [255, 255, 0]
            new_ids = s_new > sam.max()
            sub[find_boundaries(np.where(new_ids, s_new, 0), mode="inner")] = [0, 255, 0]
            Image.fromarray(sub).save(roi / "overlay_sp_qc.png")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True); df.to_csv(RESULTS / "qc_check.csv", index=False)
    print(f"\nmean of {len(df)} ROIs: doublet P {df.doublet_p.mean():.2f} R {df.doublet_r.mean():.2f} | missed P {df.missed_p.mean():.2f} R {df.missed_r.mean():.2f} | "
          f"empty P {df.empty_p.mean():.2f} R {df.empty_r.mean():.2f} | cell F1 {df.f1_before.mean():.3f} -> full {df.f1_full.mean():.3f} "
          f"(add-only {df.f1_add_only.mean():.3f}, split-only {df.f1_split_only.mean():.3f}); recall {df.r_before.mean():.3f} -> {df.r_full.mean():.3f}; precision {df.p_before.mean():.3f} -> {df.p_full.mean():.3f}")


if __name__ == "__main__":
    main()
