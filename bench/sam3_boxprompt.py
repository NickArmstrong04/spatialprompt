"""Whole-cell masks by box-prompting SAM 3's interactive (SAM-1-task) head from text seeds.

For every seed (a SAM 3 "cell" detection, which is nucleus-sized), build a box = the
seed's bounding box dilated by --dilate-um on each side, and ask the interactive head
for one mask per box (`model.predict_inst(..., box=(N, 4) xyxy, multimask_output=False)`).
Boxes are batched per 512 px tile; a seed is prompted from the tile whose core holds its
centroid. Conflicts between returned masks are resolved by nearest seed centroid.

This is also the primitive the QC agent will use for repairs (re-prompt one cell with a
corrected box). Writes data/roi/<roi>/cells_sam3box_<tag>.tif + overlay, appends to
results/sam3_wholecell.csv with config prefix "box_".

Usage: <spatial python> -X utf8 bench/sam3_boxprompt.py [--seeds dapi_boundary_cell_t512_o64_c0.3] [--dilate-um 2] [--only roi01]
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
sys.path.insert(0, str(ROOT / "bench"))
from cellpose_baseline import match_f1  # noqa: E402
from sam3_tiled import make_composite, tile_grid, tile_cores  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="dapi_boundary_cell_t512_o64_c0.3")
    ap.add_argument("--composite", default="dapi_boundary")
    ap.add_argument("--dilate-um", type=float, default=2.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    tag = f"box_{args.seeds}_{args.composite}_d{args.dilate_um:g}"

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(device="cuda" if torch.cuda.is_available() else "cpu", enable_inst_interactivity=True)
    proc = Sam3Processor(model)

    cp = pd.read_csv(RESULTS / "baseline_cellpose.csv").set_index("roi")
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / f"cells_sam3_{args.seeds}.tif").exists())
    if args.only:
        rois = [p for p in rois if p.name in args.only]
    rows = []
    for roi in rois:
        t0 = time.time()
        img = tifffile.imread(roi / "image.tif")
        rgb = make_composite(img, args.composite)
        seeds = tifffile.imread(roi / f"cells_sam3_{args.seeds}.tif").astype(np.int64)
        gt_c = tifffile.imread(roi / "cells_10x.tif").astype(np.int64)
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        d = int(round(args.dilate_um / px))
        H, W = seeds.shape
        objs = ndi.find_objects(seeds)
        cents = ndi.center_of_mass(seeds > 0, seeds, index=np.arange(1, seeds.max() + 1))
        ys, xs = tile_grid(H, args.tile, args.overlap), tile_grid(W, args.tile, args.overlap)
        ycores, xcores = tile_cores(ys, args.tile, H), tile_cores(xs, args.tile, W)
        claim = np.zeros((H, W), dtype=np.int32)
        multi = np.zeros((H, W), dtype=bool)
        n_prompted = 0
        for y0, (cy0, cy1) in zip(ys, ycores):
            for x0, (cx0, cx1) in zip(xs, xcores):
                ids = [i + 1 for i, (cy, cx) in enumerate(cents) if objs[i] is not None and cy0 <= cy < cy1 and cx0 <= cx < cx1]
                if not ids:
                    continue
                boxes = []
                for i in ids:
                    sy, sx = objs[i - 1]
                    boxes.append([max(sx.start - d, x0) - x0, max(sy.start - d, y0) - y0,
                                  min(sx.stop + d, x0 + args.tile) - x0, min(sy.stop + d, y0 + args.tile) - y0])
                boxes = np.array(boxes, dtype=np.float32)
                state = proc.set_image(Image.fromarray(rgb[y0 : y0 + args.tile, x0 : x0 + args.tile]))
                for b0 in range(0, len(ids), args.batch):
                    bb = boxes[b0 : b0 + args.batch]
                    masks, scores, _ = model.predict_inst(state, point_coords=None, point_labels=None, box=bb, multimask_output=False)
                    masks = np.asarray(masks).reshape(len(bb), -1, args.tile, args.tile)[:, 0] > 0
                    for m, i in zip(masks, ids[b0 : b0 + args.batch]):
                        f = np.zeros((H, W), dtype=bool); f[y0 : y0 + args.tile, x0 : x0 + args.tile] = m
                        multi |= f & (claim > 0)
                        claim[f & (claim == 0)] = i
                        n_prompted += 1
        # contested pixels -> nearest seed centroid among seeds that cover them is expensive;
        # approximate with nearest centroid among all seeds via a Voronoi label of seed centroids
        if multi.any():
            cy = np.array([c[0] for c in cents]); cx = np.array([c[1] for c in cents])
            marker = np.zeros((H, W), dtype=np.int32)
            marker[np.clip(cy.round().astype(int), 0, H - 1), np.clip(cx.round().astype(int), 0, W - 1)] = np.arange(1, len(cents) + 1)
            _, inds = ndi.distance_transform_edt(marker == 0, return_indices=True)
            vor = marker[inds[0], inds[1]]
            claim[multi] = vor[multi]
        # ensure every seed keeps at least its own nucleus pixels
        claim[(seeds > 0) & (claim == 0)] = seeds[(seeds > 0) & (claim == 0)]
        lab = claim.astype(np.uint32)
        m = match_f1(gt_c, lab.astype(np.int64))
        areas = np.bincount(lab.ravel())[1:]
        row = dict(roi=roi.name, config=tag, seeds=args.seeds, radius_um=args.dilate_um, sigma=np.nan, fg_q=np.nan,
                   n_cells=int(len(np.unique(lab)) - 1), n_10x=int(len(np.unique(gt_c)) - 1),
                   seed_cell_f1=match_f1(gt_c, seeds)["f1"], cell_f1=m["f1"], cell_precision=m["precision"], cell_recall=m["recall"],
                   median_area_um2=float(np.median(areas[areas > 0]) * px * px), cellpose_cell_f1=float(cp.loc[roi.name, "cell_f1"]))
        rows.append(row)
        print(f"{roi.name}: {n_prompted} boxes in {time.time()-t0:.0f}s | box-prompt F1 {m['f1']:.3f} (P {m['precision']:.2f} R {m['recall']:.2f}) "
              f"median area {row['median_area_um2']:.0f} um2 | cellpose {row['cellpose_cell_f1']:.3f}", flush=True)
        tifffile.imwrite(roi / f"cells_sam3box_{tag}.tif", lab)
        base = np.array(Image.open(roi / "composite.png"))
        y0 = x0 = (base.shape[0] - 1024) // 2
        sub = base[y0 : y0 + 1024, x0 : x0 + 1024].copy()
        sub[find_boundaries(gt_c[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [0, 255, 255]
        sub[find_boundaries(lab[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [255, 255, 0]
        Image.fromarray(sub).save(roi / f"overlay_sam3box_{tag}.png")

    df = pd.DataFrame(rows)
    out = RESULTS / "sam3_wholecell.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~((old.config == tag) & old.roi.isin(df.roi))], df], ignore_index=True)
    df.to_csv(out, index=False)
    me = df[df.config == tag]
    print(f"\n{tag}: mean cell F1 {me.cell_f1.mean():.3f} (P {me.cell_precision.mean():.2f} R {me.cell_recall.mean():.2f}), "
          f"median area {me.median_area_um2.mean():.0f} um2 | Cellpose {me.cellpose_cell_f1.mean():.3f}")


if __name__ == "__main__":
    main()
