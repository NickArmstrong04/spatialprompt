"""Tiled SAM 3 text-prompted segmentation over whole ROIs, scored against 10x and Cellpose.

Why tiles: the SAM 3 detector has num_queries=200, so a single forward pass can never
return more than 200 instances, and the processor resizes any input to 1008 px. 512 px
tiles (~150 Xenium cells) upsampled 2x work; 1024 px tiles do not (see CLAUDE.md).

Stitching: tiles overlap by --overlap px. Each tile "owns" its core region (the tile
minus half the overlap on each interior side); an instance is kept only if its centroid
falls in the owning tile's core, so each cell is predicted by exactly one tile. Kept
instances are painted into one label image in ascending score order (higher scores win
pixel conflicts).

Per ROI writes data/roi/<roi>/cells_sam3.tif (uint32), overlay_sam3.png (cyan = 10x,
yellow = SAM) and appends to results/sam3_tiled.csv (one row per ROI x config).

Usage:
  <spatial python> -X utf8 bench/sam3_tiled.py                       # all ROIs, defaults
  <spatial python> -X utf8 bench/sam3_tiled.py --only roi01 --prompt cell --tile 512 --overlap 64
  <spatial python> -X utf8 bench/sam3_tiled.py --composite dapi      # grayscale DAPI input
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
sys.path.insert(0, str(ROOT / "bench"))
from cellpose_baseline import match_f1  # noqa: E402


def norm8(ch: np.ndarray, lo_q: float = 1, hi_q: float = 99.8) -> np.ndarray:
    lo, hi = np.percentile(ch, [lo_q, hi_q])
    return (np.clip((ch.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def make_composite(img: np.ndarray, kind: str) -> np.ndarray:
    """img is (4, H, W): DAPI, boundary (ATP1A1/CD45/E-Cad), 18S, aSMA/Vim -> (H, W, 3) uint8."""
    dapi, bnd, rna, sma = (norm8(img[i]) for i in range(4))
    zero = np.zeros_like(dapi)
    if kind == "rgb":            # same as composite.png: R boundary, G 18S, B DAPI
        return np.stack([bnd, rna, dapi], -1)
    if kind == "dapi":           # grayscale nuclei
        return np.stack([dapi, dapi, dapi], -1)
    if kind == "dapi_boundary":  # R boundary, B DAPI
        return np.stack([bnd, zero, dapi], -1)
    if kind == "boundary":       # grayscale membrane
        return np.stack([bnd, bnd, bnd], -1)
    if kind == "all":            # R boundary, G 18S+aSMA, B DAPI
        return np.stack([bnd, np.maximum(rna, sma), dapi], -1)
    raise ValueError(kind)


def tile_grid(size: int, tile: int, overlap: int) -> list[int]:
    """Start offsets so tiles of `tile` px with `overlap` cover [0, size)."""
    stride = tile - overlap
    starts = list(range(0, max(size - tile, 0) + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def stitch(kept: list, H: int, W: int, tile: int, dup_cover: float = 0.5, dup_size: float = 0.5, min_area: int = 30) -> tuple[np.ndarray, int]:
    """Resolve overlapping instances from neighbouring tiles / multi-scale detections.

    Instances are visited by descending score. An instance A is a *duplicate* (dropped)
    if >= dup_cover of its pixels already belong to one accepted instance B and A is at
    least dup_size of B's area. A much smaller instance nested inside B (a single nucleus
    inside a blob that covers two) is kept, and contested pixels are then assigned to the
    instance whose centroid is nearest. Each survivor is reduced to its largest connected
    component with holes filled; instances under min_area px are dropped. The previous
    stitcher painted higher scores over lower ones and left 21 % of instances fragmented.
    """
    from scipy import ndimage as ndi
    from skimage.measure import label as cc_label

    if not kept:
        return np.zeros((H, W), dtype=np.uint32), 0
    kept = sorted(kept, key=lambda k: -k[0])
    occ = np.zeros((H, W), dtype=np.int32)       # first claimant
    multi = np.zeros((H, W), dtype=bool)         # contested pixels
    inst = []                                    # (area, cy, cx, y0, x0, mask) of accepted
    areas = []
    for s, m, y0, x0 in kept:
        a = int(m.sum())
        if a < min_area:
            continue
        sub = occ[y0 : y0 + tile, x0 : x0 + tile]
        ov = sub[m]
        if ov.any():
            counts = np.bincount(ov[ov > 0])
            b = int(np.argmax(counts)); top = counts[b]
            if top / a >= dup_cover and a >= dup_size * areas[b - 1]:
                continue
        k = len(inst) + 1
        yy, xx = np.nonzero(m)
        inst.append((a, yy.mean() + y0, xx.mean() + x0, y0, x0, m)); areas.append(a)
        multi[y0 : y0 + tile, x0 : x0 + tile] |= m & (sub > 0)
        sub[m & (sub == 0)] = k
    n_dedup = len(inst)
    # contested pixels -> nearest centroid among claimants (evaluated locally per instance)
    if multi.any():
        bestd = np.full((H, W), np.inf, dtype=np.float32)
        yy_g, xx_g = np.mgrid[0:tile, 0:tile]
        for k, (a, cy, cx, y0, x0, m) in enumerate(inst, start=1):
            win = (slice(y0, y0 + tile), slice(x0, x0 + tile))
            c = m & multi[win]
            if not c.any():
                continue
            d = (yy_g + y0 - cy) ** 2 + (xx_g + x0 - cx) ** 2
            better = c & (d < bestd[win])
            bestd[win][better] = d[better]
            occ[win][better] = k
    # clean each instance
    out = np.zeros((H, W), dtype=np.uint32); n = 0
    for k, sl in enumerate(ndi.find_objects(occ), start=1):
        if sl is None:
            continue
        sub = occ[sl] == k
        cc = cc_label(sub)
        if cc.max() > 1:
            sizes = np.bincount(cc.ravel())[1:]; sub = cc == (np.argmax(sizes) + 1)
        sub = ndi.binary_fill_holes(sub)
        if sub.sum() < min_area:
            continue
        n += 1
        tgt = out[sl]; tgt[sub & (tgt == 0)] = n
    return out, n_dedup


def tile_cores(starts: list[int], tile: int, size: int) -> list[tuple[int, int]]:
    """Ownership interval [c0, c1) for each tile start: midpoints between neighbouring tile
    centres, so cores partition [0, size) even when the last tile overlaps more."""
    centres = [s + tile / 2 for s in starts]
    cores = []
    for i, s in enumerate(starts):
        c0 = 0 if i == 0 else int(round((centres[i - 1] + centres[i]) / 2))
        c1 = size if i == len(starts) - 1 else int(round((centres[i] + centres[i + 1]) / 2))
        cores.append((c0, c1))
    return cores


def run_roi(proc, rgb: np.ndarray, prompt: str, tile: int, overlap: int) -> tuple[np.ndarray, dict]:
    """Returns stitched uint32 label image and stats."""
    H, W = rgb.shape[:2]
    ys, xs = tile_grid(H, tile, overlap), tile_grid(W, tile, overlap)
    ycores, xcores = tile_cores(ys, tile, H), tile_cores(xs, tile, W)
    kept: list[tuple[float, np.ndarray, int, int]] = []  # (score, mask, y0, x0)
    n_raw = 0; per_tile = []
    t0 = time.time()
    for y0, (gy0, gy1) in zip(ys, ycores):
        for x0, (gx0, gx1) in zip(xs, xcores):
            crop = rgb[y0 : y0 + tile, x0 : x0 + tile]
            state = proc.set_image(Image.fromarray(crop))
            st = proc.set_text_prompt(prompt, state)
            masks = st["masks"].detach().float().cpu().numpy().reshape(-1, tile, tile) > 0.5
            scores = st["scores"].detach().float().cpu().numpy().ravel()
            n_raw += len(masks); per_tile.append(len(masks))
            # core region of this tile in tile coordinates
            cy0, cy1, cx0, cx1 = gy0 - y0, gy1 - y0, gx0 - x0, gx1 - x0
            for m, s in zip(masks, scores):
                if not m.any():
                    continue
                yy, xx = np.nonzero(m)
                cy, cx = yy.mean(), xx.mean()
                if cy0 <= cy < cy1 and cx0 <= cx < cx1:
                    kept.append((float(s), m, y0, x0))
    lab, n_dedup = stitch(kept, H, W, tile)
    stats = dict(n_tiles=len(ys) * len(xs), n_raw=n_raw, n_kept=len(kept), n_dedup=n_dedup, n_final=int(lab.max()),
                 max_per_tile=int(max(per_tile)), tiles_ge190=int(sum(v >= 190 for v in per_tile)), sec=time.time() - t0)
    return lab, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--prompt", default="cell")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--composite", default="rgb", choices=["rgb", "dapi", "dapi_boundary", "boundary", "all"])
    ap.add_argument("--tag", default="", help="suffix for output files (default from config)")
    args = ap.parse_args()

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(device="cuda" if torch.cuda.is_available() else "cpu")
    proc = Sam3Processor(model, confidence_threshold=args.threshold)
    tag = args.tag or f"{args.composite}_{args.prompt.replace(' ', '-')}_t{args.tile}_o{args.overlap}_c{args.threshold}"
    print("config", tag, flush=True)

    cp = None
    if (RESULTS / "baseline_cellpose.csv").exists():
        cp = pd.read_csv(RESULTS / "baseline_cellpose.csv").set_index("roi")

    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / "image.tif").exists())
    if args.only:
        rois = [p for p in rois if p.name in args.only]
    rows = []
    for roi in rois:
        img = tifffile.imread(roi / "image.tif")
        rgb = make_composite(img, args.composite)
        gt_c = tifffile.imread(roi / "cells_10x.tif").astype(np.int64)
        gt_n = tifffile.imread(roi / "nuclei_10x.tif").astype(np.int64)
        lab, st = run_roi(proc, rgb, args.prompt, args.tile, args.overlap)
        mc = match_f1(gt_c, lab.astype(np.int64))
        mn = match_f1(gt_n, lab.astype(np.int64))
        areas = np.bincount(lab.ravel())[1:]
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        row = dict(roi=roi.name, config=tag, prompt=args.prompt, composite=args.composite, tile=args.tile,
                   overlap=args.overlap, threshold=args.threshold, **st,
                   n_10x_cells=int(len(np.unique(gt_c)) - 1), n_10x_nuclei=int(len(np.unique(gt_n)) - 1),
                   cell_f1=mc["f1"], cell_precision=mc["precision"], cell_recall=mc["recall"],
                   nuc_f1=mn["f1"], nuc_precision=mn["precision"], nuc_recall=mn["recall"],
                   median_area_um2=float(np.median(areas[areas > 0]) * px * px) if len(areas) else float("nan"),
                   cellpose_cell_f1=float(cp.loc[roi.name, "cell_f1"]) if cp is not None and roi.name in cp.index else float("nan"),
                   cellpose_nuc_f1=float(cp.loc[roi.name, "nuc_f1"]) if cp is not None and roi.name in cp.index else float("nan"))
        rows.append(row)
        print(f"{roi.name}: {st['n_tiles']} tiles (max/tile {st['max_per_tile']}, {st['tiles_ge190']} at cap), {st['n_raw']} raw -> {st['n_final']} cells ({st['sec']:.0f}s) | "
              f"vs 10x cells ({row['n_10x_cells']}) F1 {mc['f1']:.3f} P {mc['precision']:.2f} R {mc['recall']:.2f} | "
              f"vs nuclei F1 {mn['f1']:.3f} | cellpose cell F1 {row['cellpose_cell_f1']:.3f}", flush=True)

        tifffile.imwrite(roi / f"cells_sam3_{tag}.tif", lab)
        if tag.startswith("rgb_cell_t512_o64"):  # canonical config also written under a stable name
            tifffile.imwrite(roi / "cells_sam3.tif", lab)
        base = np.array(Image.open(roi / "composite.png")) if args.composite == "rgb" else rgb
        y0 = x0 = (base.shape[0] - 1024) // 2
        sub = base[y0 : y0 + 1024, x0 : x0 + 1024].copy()
        sub[find_boundaries(gt_c[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [0, 255, 255]
        sub[find_boundaries(lab[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [255, 255, 0]
        Image.fromarray(sub).save(roi / f"overlay_sam3_{tag}.png")

    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    out = RESULTS / "sam3_tiled.csv"
    if out.exists():
        old = pd.read_csv(out)
        old = old[~((old.config == tag) & old.roi.isin(df.roi))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    me = df[df.config == tag]
    print(f"\n{tag}: mean cell F1 {me.cell_f1.mean():.3f} (P {me.cell_precision.mean():.2f} R {me.cell_recall.mean():.2f}) | "
          f"mean nucleus F1 {me.nuc_f1.mean():.3f} | Cellpose cell F1 {me.cellpose_cell_f1.mean():.3f}  -> {out}")


if __name__ == "__main__":
    main()
