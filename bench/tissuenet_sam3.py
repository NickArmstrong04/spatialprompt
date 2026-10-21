"""TissueNet benchmark: the packaged SAM 3 pipeline (text seeds -> box-prompted cells) vs human labels.

Same harness as bench/tissuenet_cellpose.py (same split loader, same metric), so the two
results CSVs line up row for row. TissueNet tiles are 256 px, so one tile = one image
(no stitching); the composite is R = whole-cell/membrane channel, B = nuclear channel, the
same layout that worked on Xenium. Pixel size comes from the v1.1 meta per image.

Usage: <spatial python> -X utf8 bench/tissuenet_sam3.py [--n 0] [--tile 256] [--dilate-um 2] [--multimask]
Outputs results/tissuenet_sam3_<split>.csv and prints mean F1 overall and per tissue.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "bench")); sys.path.insert(0, str(ROOT / "src"))
from tissuenet_cellpose import load_split  # noqa: E402
from spatialprompt import Segmenter, SegmentConfig, match_f1  # noqa: E402
from spatialprompt.composite import rgb_from_channels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0, help="images to score (0 = all)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--default-px", type=float, default=0.5, help="um/px when meta has no pixel_size")
    for k, v in asdict(SegmentConfig()).items():
        if k == "pixel_size_um":
            continue
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v if k != "tile" else 256)
    a = ap.parse_args()
    tag = a.tag or (f"t{a.tile}_c{a.threshold:g}_d{a.dilate_um:g}" + ("_mm" if a.multimask else "") + ("" if a.expand else "_seeds"))

    X, y, meta, idx = load_split(a.split, a.n or None)
    has_meta = isinstance(meta, pd.DataFrame)
    seg = Segmenter(threshold=a.threshold)
    print(f"config {tag} | model loaded in {seg.load_sec:.0f}s", flush=True)
    rows = []
    t0 = time.time()
    for k in range(len(X)):
        nuc, mem = X[k, ..., 0], X[k, ..., 1]
        px = a.default_px
        if has_meta:
            try:
                px = float(meta.pixel_size.iloc[k])
            except (TypeError, ValueError):
                pass
        cfg = SegmentConfig.from_dict({**vars(a), "pixel_size_um": px})
        res = seg.segment(rgb_from_channels(nuc, mem), cfg)
        gt_c, gt_n = y[k, ..., 0].astype(np.int64), y[k, ..., 1].astype(np.int64)
        mc, mn = match_f1(gt_c, res.cells), match_f1(gt_n, res.nuclei)
        row = dict(index=int(idx[k]), config=tag, n_gt=int(len(np.unique(gt_c)) - 1), n_pred=int(res.stats.get("n_cells", res.stats["n_seeds"])),
                   n_seeds=res.stats["n_seeds"], max_per_tile=res.stats["max_per_tile"], pixel_size_um=px, **mc,
                   nuc_f1=mn["f1"], nuc_precision=mn["precision"], nuc_recall=mn["recall"])
        if has_meta:
            row["tissue"] = meta.tissue.iloc[k]; row["specimen"] = meta.specimen.iloc[k]
            row["experiment"] = str(meta.experiment.iloc[k]).split("/")[-1]
        rows.append(row)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(X)}  running mean cell F1 {np.mean([r['f1'] for r in rows]):.3f} nuc {np.mean([r['nuc_f1'] for r in rows]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"tissuenet_sam3_{a.split}.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~((old.config == tag) & old["index"].isin(df["index"]))], df], ignore_index=True)
    df.to_csv(out, index=False)
    me = df[df.config == tag]
    print(f"\nTissueNet {a.split} n={len(me)} [{tag}]: SAM 3 whole-cell F1@0.5 = {me.f1.mean():.3f} "
          f"(P {me.precision.mean():.3f} R {me.recall.mean():.3f}) | nucleus F1 {me.nuc_f1.mean():.3f} | {(time.time()-t0)/len(X):.1f} s/img  wrote {out}")
    if has_meta:
        print(me.groupby("tissue")[["f1", "nuc_f1", "n_gt", "n_pred"]].mean().round(3).sort_values("f1", ascending=False).to_string())
    cp = RESULTS / f"tissuenet_cellpose_{a.split}.csv"
    if cp.exists():
        c = pd.read_csv(cp).set_index("index")
        j = me.set_index("index").join(c[["f1"]].rename(columns={"f1": "cellpose_f1"}), how="inner")
        print(f"same images: SAM {j.f1.mean():.3f} vs Cellpose {j.cellpose_f1.mean():.3f} (n={len(j)})")


if __name__ == "__main__":
    main()
