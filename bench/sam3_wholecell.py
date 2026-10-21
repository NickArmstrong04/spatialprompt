"""Grow whole-cell masks from SAM 3 text-prompt seeds by watershed into the boundary stain.

SAM 3 with the prompt "cell" returns nucleus-sized masks (see CLAUDE.md). 10x's own
multimodal segmenter does what we do here: start from a nucleus, expand outward until a
membrane (boundary stain ridge) or a fixed radius is hit. We use SAM's instances as
watershed markers on the smoothed boundary-stain image (bright membranes = ridges),
restricted to pixels within --radius-um of a seed that also show tissue signal.

Writes data/roi/<roi>/cells_sam3ws_<tag>.tif and overlay, appends to results/sam3_wholecell.csv.

Usage: <spatial python> -X utf8 bench/sam3_wholecell.py [--seeds dapi_boundary_cell_t512_o64_c0.3]
       [--radius-um 5] [--sigma 1.5] [--fg-q 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries, watershed

ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "bench"))
from cellpose_baseline import match_f1  # noqa: E402


def norm(ch: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(ch, [1, 99.8])
    return np.clip((ch.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def expand(seeds: np.ndarray, img: np.ndarray, px_um: float, radius_um: float, sigma: float, fg_q: float,
           use_interior: bool) -> np.ndarray:
    dapi, bnd, rna = norm(img[0]), norm(img[1]), norm(img[2])
    elevation = gaussian(bnd, sigma=sigma, preserve_range=True)
    # allowed region: within radius of any seed, and some tissue signal present
    dist = ndi.distance_transform_edt(seeds == 0)
    tissue = np.maximum(np.maximum(dapi, bnd), rna) if use_interior else np.maximum(dapi, bnd)
    thr = np.percentile(tissue, fg_q)
    allowed = (dist <= radius_um / px_um) & ((tissue > thr) | (seeds > 0))
    return watershed(elevation, markers=seeds.astype(np.int32), mask=allowed).astype(np.uint32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="dapi_boundary_cell_t512_o64_c0.3", help="tag of cells_sam3_<tag>.tif to use as markers")
    ap.add_argument("--radius-um", type=float, default=5.0)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--fg-q", type=float, default=30.0, help="percentile of tissue signal below which pixels are background")
    ap.add_argument("--no-interior", action="store_true", help="exclude 18S from the tissue-foreground test")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    tag = f"{args.seeds}_r{args.radius_um:g}_s{args.sigma:g}_q{args.fg_q:g}{'_noint' if args.no_interior else ''}"

    cp = pd.read_csv(RESULTS / "baseline_cellpose.csv").set_index("roi") if (RESULTS / "baseline_cellpose.csv").exists() else None
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / f"cells_sam3_{args.seeds}.tif").exists())
    if args.only:
        rois = [p for p in rois if p.name in args.only]
    rows = []
    for roi in rois:
        img = tifffile.imread(roi / "image.tif")
        seeds = tifffile.imread(roi / f"cells_sam3_{args.seeds}.tif")
        gt_c = tifffile.imread(roi / "cells_10x.tif").astype(np.int64)
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        lab = expand(seeds, img, px, args.radius_um, args.sigma, args.fg_q, not args.no_interior)
        m_seed = match_f1(gt_c, seeds.astype(np.int64))
        m = match_f1(gt_c, lab.astype(np.int64))
        areas = np.bincount(lab.ravel())[1:]
        row = dict(roi=roi.name, config=tag, seeds=args.seeds, radius_um=args.radius_um, sigma=args.sigma, fg_q=args.fg_q,
                   n_cells=int(lab.max()), n_10x=int(len(np.unique(gt_c)) - 1),
                   seed_cell_f1=m_seed["f1"], cell_f1=m["f1"], cell_precision=m["precision"], cell_recall=m["recall"],
                   median_area_um2=float(np.median(areas[areas > 0]) * px * px),
                   cellpose_cell_f1=float(cp.loc[roi.name, "cell_f1"]) if cp is not None else float("nan"))
        rows.append(row)
        print(f"{roi.name}: seeds F1 {m_seed['f1']:.3f} -> watershed F1 {m['f1']:.3f} (P {m['precision']:.2f} R {m['recall']:.2f}) "
              f"median area {row['median_area_um2']:.0f} um2 | cellpose {row['cellpose_cell_f1']:.3f}", flush=True)
        tifffile.imwrite(roi / f"cells_sam3ws_{tag}.tif", lab)
        base = np.array(Image.open(roi / "composite.png"))
        y0 = x0 = (base.shape[0] - 1024) // 2
        sub = base[y0 : y0 + 1024, x0 : x0 + 1024].copy()
        sub[find_boundaries(gt_c[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [0, 255, 255]
        sub[find_boundaries(lab[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [255, 255, 0]
        Image.fromarray(sub).save(roi / f"overlay_sam3ws_{tag}.png")

    df = pd.DataFrame(rows)
    out = RESULTS / "sam3_wholecell.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~((old.config == tag) & old.roi.isin(df.roi))], df], ignore_index=True)
    df.to_csv(out, index=False)
    me = df[df.config == tag]
    print(f"\n{tag}: mean cell F1 {me.cell_f1.mean():.3f} (P {me.cell_precision.mean():.2f} R {me.cell_recall.mean():.2f}), "
          f"median area {me.median_area_um2.mean():.0f} um2 | seeds {me.seed_cell_f1.mean():.3f} | Cellpose {me.cellpose_cell_f1.mean():.3f}")


if __name__ == "__main__":
    main()
