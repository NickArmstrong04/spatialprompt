"""Baseline segmentation on the Xenium ROIs: Cellpose (cpsam) vs 10x's multimodal masks.

For every ROI in data/roi/*/ this runs Cellpose on [boundary stain, DAPI] (whole-cell) and
on DAPI alone (nuclei), writes the label images, and scores agreement with 10x's own cell
and nucleus masks using matched-instance F1 at IoU 0.5 (the TissueNet/Mesmer protocol).

NB: 10x's masks are a *reference*, not ground truth. The number to quote in the write-up
is agreement between two automated methods; human-annotated F1 comes from TissueNet.

Outputs: data/roi/<roi>/cells_cellpose.tif, nuclei_cellpose.tif, overlay_cellpose.png
         results/baseline_cellpose.csv

Usage: <spatial env python> -X utf8 bench/cellpose_baseline.py [--only roi01] [--diameter 0]
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
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"


# match_f1 lives in the package now (src/spatialprompt/metrics.py); re-exported here because the
# other bench scripts import it from this module.
sys.path.insert(0, str(ROOT / "src"))
from spatialprompt.metrics import match_f1  # noqa: E402,F401


def norm(ch: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(ch, [1, 99.8])
    return np.clip((ch.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--diameter", type=float, default=0.0, help="0 = let Cellpose estimate")
    ap.add_argument("--model", default="cpsam")
    args = ap.parse_args()

    import torch
    from cellpose import models

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, "| cellpose model", args.model)
    model = models.CellposeModel(gpu=dev.type == "cuda", pretrained_model=args.model)

    rows = []
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / "image.tif").exists())
    if args.only:
        rois = [p for p in rois if p.name in args.only]
    for roi in rois:
        img = tifffile.imread(roi / "image.tif")  # (4, H, W): DAPI, boundary, 18S, aSMA/Vim
        gt_c = tifffile.imread(roi / "cells_10x.tif")
        gt_n = tifffile.imread(roi / "nuclei_10x.tif")
        meta = json.loads((roi / "roi.json").read_text())
        px = meta["pixel_size_um"]

        # whole cell: boundary stain as cytoplasm/membrane channel, DAPI as nucleus channel
        cell_in = np.stack([norm(img[1]), norm(img[0]), np.zeros_like(img[0], dtype=np.float32)], -1)
        t0 = time.time()
        masks_c, _, _ = model.eval(cell_in, diameter=args.diameter or None, channel_axis=-1, flow_threshold=0.4, cellprob_threshold=0.0)
        t_cell = time.time() - t0
        # nuclei: DAPI only
        nuc_in = np.stack([norm(img[0]), np.zeros_like(img[0], dtype=np.float32), np.zeros_like(img[0], dtype=np.float32)], -1)
        t0 = time.time()
        masks_n, _, _ = model.eval(nuc_in, diameter=args.diameter or None, channel_axis=-1, flow_threshold=0.4, cellprob_threshold=0.0)
        t_nuc = time.time() - t0

        masks_c = masks_c.astype(np.uint32); masks_n = masks_n.astype(np.uint32)
        tifffile.imwrite(roi / "cells_cellpose.tif", masks_c)
        tifffile.imwrite(roi / "nuclei_cellpose.tif", masks_n)

        mc = match_f1(gt_c, masks_c); mn = match_f1(gt_n, masks_n)
        n_c, n_n = int(masks_c.max()), int(masks_n.max())
        row = dict(roi=roi.name, n_10x_cells=int(len(np.unique(gt_c)) - 1), n_cellpose_cells=n_c,
                   cell_f1=mc["f1"], cell_precision=mc["precision"], cell_recall=mc["recall"],
                   n_10x_nuclei=int(len(np.unique(gt_n)) - 1), n_cellpose_nuclei=n_n,
                   nuc_f1=mn["f1"], nuc_precision=mn["precision"], nuc_recall=mn["recall"],
                   median_cell_area_um2_10x=float(np.median(np.bincount(gt_c.ravel())[1:][np.bincount(gt_c.ravel())[1:] > 0]) * px * px),
                   median_cell_area_um2_cp=float(np.median(np.bincount(masks_c.ravel())[1:][np.bincount(masks_c.ravel())[1:] > 0]) * px * px) if n_c else float("nan"),
                   sec_cell=t_cell, sec_nuc=t_nuc)
        rows.append(row)
        print(f"{roi.name}: cells 10x={row['n_10x_cells']} cp={n_c} F1@0.5={mc['f1']:.3f} (P {mc['precision']:.2f} R {mc['recall']:.2f}) | "
              f"nuclei 10x={row['n_10x_nuclei']} cp={n_n} F1@0.5={mn['f1']:.3f} | {t_cell:.0f}s + {t_nuc:.0f}s", flush=True)

        # overlay: composite + cellpose (yellow) + 10x (cyan) boundaries, central 1024 crop, 2x
        rgb = np.array(Image.open(roi / "composite.png"))
        y0 = x0 = (rgb.shape[0] - 1024) // 2
        sub = rgb[y0:y0 + 1024, x0:x0 + 1024].copy()
        sub[find_boundaries(gt_c[y0:y0 + 1024, x0:x0 + 1024], mode="inner")] = [0, 255, 255]
        sub[find_boundaries(masks_c[y0:y0 + 1024, x0:x0 + 1024], mode="inner")] = [255, 255, 0]
        Image.fromarray(sub).save(roi / "overlay_cellpose.png")

    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "baseline_cellpose.csv", index=False)
    print("\nmean over ROIs: cell F1 %.3f  nucleus F1 %.3f" % (df.cell_f1.mean(), df.nuc_f1.mean()))
    print("wrote", RESULTS / "baseline_cellpose.csv")


if __name__ == "__main__":
    main()
