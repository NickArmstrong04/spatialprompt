"""Run the packaged pipeline (spatialprompt.segment.Segmenter) on the Xenium ROIs and score it.

This is the regression check for src/spatialprompt/segment.py: with defaults it must reproduce
bench/sam3_tiled.py (384 px seeds, nucleus F1 ~0.81) + bench/sam3_boxprompt.py (2 um boxes,
cell F1 ~0.73). Any SegmentConfig field can be overridden on the CLI to sweep variants.

Writes data/roi/<roi>/cells_sp_<tag>.tif, nuclei_sp_<tag>.tif, overlay_sp_<tag>.png and appends
one row per ROI to results/sam3_pipeline.csv.

Usage: <spatial python> -X utf8 bench/sam3_pipeline.py [--only roi01] [--multimask] [--dilate-um 2] [--tag name]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from spatialprompt import Segmenter, SegmentConfig, make_composite, match_f1  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--composite", default="dapi_boundary")
    ap.add_argument("--tag", default="")
    for k, v in asdict(SegmentConfig()).items():
        if k == "pixel_size_um":
            continue
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()
    tag = a.tag or (f"{a.composite}_{a.prompt.replace(' ', '-')}_t{a.tile}_c{a.threshold:g}_d{a.dilate_um:g}"
                    + ("_mm" if a.multimask else "") + ("_hint" if a.seed_hint else "") + ("" if a.expand else "_seeds"))
    print("config", tag, flush=True)

    seg = Segmenter(threshold=a.threshold)
    print(f"model loaded in {seg.load_sec:.0f}s", flush=True)
    cp = pd.read_csv(RESULTS / "baseline_cellpose.csv").set_index("roi") if (RESULTS / "baseline_cellpose.csv").exists() else None
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / "image.tif").exists())
    if a.only:
        rois = [p for p in rois if p.name in a.only]
    rows = []
    for roi in rois:
        t0 = time.time()
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        cfg = SegmentConfig.from_dict({**vars(a), "pixel_size_um": px})
        rgb = make_composite(tifffile.imread(roi / "image.tif"), a.composite)
        res = seg.segment(rgb, cfg)
        gt_c = tifffile.imread(roi / "cells_10x.tif").astype(np.int64)
        gt_n = tifffile.imread(roi / "nuclei_10x.tif").astype(np.int64)
        mc, mn = match_f1(gt_c, res.cells), match_f1(gt_n, res.nuclei)
        areas = np.bincount(res.cells.ravel())[1:]
        st = {k: v for k, v in res.stats.items() if k != "config"}
        row = dict(roi=roi.name, config=tag, composite=a.composite, **{k: v for k, v in asdict(cfg).items()}, **st,
                   n_10x_cells=int(len(np.unique(gt_c)) - 1), n_10x_nuclei=int(len(np.unique(gt_n)) - 1),
                   cell_f1=mc["f1"], cell_precision=mc["precision"], cell_recall=mc["recall"],
                   nuc_f1=mn["f1"], nuc_precision=mn["precision"], nuc_recall=mn["recall"],
                   median_area_um2=float(np.median(areas[areas > 0]) * px * px) if (areas > 0).any() else float("nan"),
                   cellpose_cell_f1=float(cp.loc[roi.name, "cell_f1"]) if cp is not None and roi.name in cp.index else float("nan"),
                   cellpose_nuc_f1=float(cp.loc[roi.name, "nuc_f1"]) if cp is not None and roi.name in cp.index else float("nan"),
                   total_sec=time.time() - t0)
        rows.append(row)
        print(f"{roi.name}: {st['n_seeds']} seeds ({st['detect_sec']:.0f}s)" + (f" -> {st['n_cells']} cells ({st['expand_sec']:.0f}s, pred IoU {st['mean_pred_iou']:.2f})" if cfg.expand else "")
              + f" | cell F1 {mc['f1']:.3f} (P {mc['precision']:.2f} R {mc['recall']:.2f}) nuc F1 {mn['f1']:.3f} | cellpose {row['cellpose_cell_f1']:.3f}", flush=True)
        tifffile.imwrite(roi / f"cells_sp_{tag}.tif", res.cells)
        tifffile.imwrite(roi / f"nuclei_sp_{tag}.tif", res.nuclei)
        base = np.array(Image.open(roi / "composite.png"))
        y0 = x0 = (base.shape[0] - 1024) // 2
        sub = base[y0 : y0 + 1024, x0 : x0 + 1024].copy()
        sub[find_boundaries(gt_c[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [0, 255, 255]
        sub[find_boundaries(res.cells[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [255, 255, 0]
        Image.fromarray(sub).save(roi / f"overlay_sp_{tag}.png")

    df = pd.DataFrame(rows)
    out = RESULTS / "sam3_pipeline.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~((old.config == tag) & old.roi.isin(df.roi))], df], ignore_index=True)
    df.to_csv(out, index=False)
    me = df[df.config == tag]
    print(f"\n{tag}: mean cell F1 {me.cell_f1.mean():.3f} (P {me.cell_precision.mean():.2f} R {me.cell_recall.mean():.2f}) | "
          f"nucleus F1 {me.nuc_f1.mean():.3f} | {me.total_sec.mean():.0f} s/ROI | Cellpose cell {me.cellpose_cell_f1.mean():.3f} nuc {me.cellpose_nuc_f1.mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
