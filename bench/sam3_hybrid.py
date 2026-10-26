"""Hybrid seeds: can better seeds push SAM's box-prompted whole cells past Cellpose?

SAM's whole-cell F1 vs 10x (0.725) is limited by the text-prompt seed pass (recall), not by the
box-prompt outlines (ceiling tests in CLAUDE.md). Here the same `Segmenter.expand` is run from
four seed sets and scored identically:

  sam           SAM text seeds (control; must reproduce 0.725)
  cellpose_nuc  Cellpose nuclei (cpsam on DAPI), relabelled 1..n
  union         SAM seeds + the Cellpose nuclei whose centroid is not inside any SAM seed dilated by margin_um (1 um)
  union_m2      same with margin 2 um

Writes results/sam3_hybrid.csv; per ROI cells_sp_hybrid.tif / nuclei_sp_hybrid.tif for --save
(default union); overlay_sp_hybrid.png for roi01 (cyan = 10x, yellow = hybrid, green = cells
whose seed came from Cellpose). No API calls.

Usage: <spatial python> -X utf8 bench/sam3_hybrid.py [--only roi01] [--variants sam union] [--save union]
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
from skimage.segmentation import find_boundaries, relabel_sequential

ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from spatialprompt import Segmenter, make_composite, match_f1  # noqa: E402

TAG = "dapi_boundary_cell_t384_c0.3_d2"


def union_seeds(a: np.ndarray, b: np.ndarray, pixel_size_um: float, margin_um: float = 1.0) -> tuple[np.ndarray, int]:
    """a's seeds plus those of b whose centroid lies outside every a-seed dilated by margin_um.
    b ids are offset past a.max(); where both overlap, a's pixels win. Returns (seeds, n_added)."""
    a = np.asarray(a).astype(np.int64); b = np.asarray(b).astype(np.int64)
    ids = np.unique(b); ids = ids[ids > 0]
    if len(ids) == 0:
        return a.copy(), 0
    m = max(int(round(margin_um / pixel_size_um)), 1)
    zone = ndi.binary_dilation(a > 0, iterations=m)
    cents = np.array(ndi.center_of_mass(b > 0, b, index=ids), dtype=float)
    yi = np.clip(cents[:, 0].round().astype(int), 0, b.shape[0] - 1); xi = np.clip(cents[:, 1].round().astype(int), 0, b.shape[1] - 1)
    add = ids[~zone[yi, xi]]
    out = a.copy()
    sel = np.isin(b, add) & (a == 0)
    out[sel] = b[sel] + int(a.max())
    return out, int(len(add))


def seed_sets(sam: np.ndarray, cp: np.ndarray, px: float) -> dict[str, tuple[np.ndarray, int]]:
    cp_seq = relabel_sequential(cp)[0].astype(np.int64)
    u1, n1 = union_seeds(sam, cp, px, 1.0)
    u2, n2 = union_seeds(sam, cp, px, 2.0)
    return {"sam": (sam, 0), "cellpose_nuc": (cp_seq, int(cp_seq.max())), "union": (u1, n1), "union_m2": (u2, n2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--variants", nargs="*", default=["sam", "cellpose_nuc", "union", "union_m2"])
    ap.add_argument("--save", default="union", help="variant written to cells_sp_hybrid.tif / nuclei_sp_hybrid.tif")
    ap.add_argument("--dilate-um", type=float, default=2.0)
    a = ap.parse_args()
    seg = Segmenter()
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / f"nuclei_sp_{TAG}.tif").exists() and (p / "nuclei_cellpose.tif").exists())
    if a.only:
        rois = [p for p in rois if p.name in a.only]
    rows = []
    for roi in rois:
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        img = tifffile.imread(roi / "image.tif"); rgb = make_composite(img, "dapi_boundary")
        gt_c = tifffile.imread(roi / "cells_10x.tif").astype(np.int64); gt_n = tifffile.imread(roi / "nuclei_10x.tif").astype(np.int64)
        sam = tifffile.imread(roi / f"nuclei_sp_{TAG}.tif").astype(np.int64)
        cp = tifffile.imread(roi / "nuclei_cellpose.tif").astype(np.int64)
        sets = seed_sets(sam, cp, px)
        for v in a.variants:
            seeds, n_added = sets[v]
            t0 = time.time()
            cells, st = seg.expand(rgb, seeds, dilate_um=a.dilate_um, pixel_size_um=px)
            cells = cells.astype(np.int64)
            mc = match_f1(gt_c, cells); mn = match_f1(gt_n, seeds)
            row = dict(roi=roi.name, variant=v, n_seeds=int(len(np.unique(seeds)) - 1), n_added=n_added, n_cells=int(len(np.unique(cells)) - 1),
                       cell_f1=mc["f1"], cell_p=mc["precision"], cell_r=mc["recall"], nuc_f1=mn["f1"], nuc_p=mn["precision"], nuc_r=mn["recall"],
                       mean_pred_iou=st.get("mean_pred_iou"), expand_sec=time.time() - t0)
            rows.append(row)
            print(f"{roi.name}/{v}: seeds {row['n_seeds']} (+{n_added}) -> cells {row['n_cells']} | cell F1 {mc['f1']:.3f} (P {mc['precision']:.2f} R {mc['recall']:.2f}) "
                  f"| nucleus F1 {mn['f1']:.3f} | {row['expand_sec']:.0f}s", flush=True)
            if v == a.save:
                tifffile.imwrite(roi / "cells_sp_hybrid.tif", cells.astype(np.uint32)); tifffile.imwrite(roi / "nuclei_sp_hybrid.tif", seeds.astype(np.uint32))
                if roi.name == "roi01":
                    base = np.array(Image.open(roi / "composite.png").convert("RGB")); y0 = x0 = (base.shape[0] - 1024) // 2
                    sub = base[y0 : y0 + 1024, x0 : x0 + 1024].copy(); c = cells[y0 : y0 + 1024, x0 : x0 + 1024]
                    sub[find_boundaries(gt_c[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [0, 255, 255]
                    sub[find_boundaries(c, mode="inner")] = [255, 255, 0]
                    sub[find_boundaries(np.where(c > sam.max(), c, 0), mode="inner")] = [0, 255, 0]
                    Image.fromarray(sub).save(roi / "overlay_sp_hybrid.png")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True); df.to_csv(RESULTS / "sam3_hybrid.csv", index=False)
    g = df.groupby("variant")[["n_seeds", "n_added", "n_cells", "cell_f1", "cell_p", "cell_r", "nuc_f1", "expand_sec"]].mean().round(3)
    print("\nmean over ROIs (Cellpose whole-cell reference: 0.782):\n" + g.to_string())
    piv = df.pivot_table(index="roi", columns="variant", values="cell_f1").round(3)
    print("\ncell F1 per ROI:\n" + piv.to_string())


if __name__ == "__main__":
    main()
