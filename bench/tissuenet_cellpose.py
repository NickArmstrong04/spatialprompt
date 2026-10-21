"""TissueNet benchmark harness: Cellpose (cpsam) whole-cell F1 at IoU 0.5 vs human labels.

TissueNet v1.x (Greenwald et al. 2022) ships as npz splits with
  X: (n, 512, 512, 2) float  channel 0 = nuclear, channel 1 = whole-cell/membrane
  y: (n, 512, 512, 2) int    channel 0 = whole-cell labels, channel 1 = nuclear labels
and (v1.1) a metadata array with platform / tissue per image. This script prints the
keys it finds, so run it once with --inspect on a new download before trusting the
channel assumptions above.

Usage:
  <spatial python> -X utf8 bench/tissuenet_cellpose.py --inspect
  <spatial python> -X utf8 bench/tissuenet_cellpose.py --split test --n 100
Outputs results/tissuenet_cellpose.csv (per image) and prints the mean F1.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TN_DIR = ROOT / "data" / "tissuenet"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "bench"))
from cellpose_baseline import match_f1  # noqa: E402


def find_npz(split: str) -> Path:
    cands = sorted(TN_DIR.rglob(f"*{split}*.npz"))
    if not cands:
        zips = sorted(TN_DIR.glob("*.zip")) + sorted((Path.home() / "Downloads").glob("tissuenet*.zip"))
        if zips:
            print("unzipping", zips[0], "->", TN_DIR)
            with zipfile.ZipFile(zips[0]) as z:
                z.extractall(TN_DIR)
            cands = sorted(TN_DIR.rglob(f"*{split}*.npz"))
    if not cands:
        raise SystemExit(f"no *{split}*.npz under {TN_DIR}; put the TissueNet zip or npz files there")
    return cands[0]


def load_split(split: str, n: int | None, seed: int = 0):
    p = find_npz(split)
    d = np.load(p, allow_pickle=True)
    print("file", p.name, "| keys", list(d.keys()), "| X", d["X"].shape, d["X"].dtype, "| y", d["y"].shape, d["y"].dtype)
    X, y = d["X"], d["y"]
    meta = d["meta"] if "meta" in d else None
    if meta is not None and meta.ndim == 2 and len(meta) > len(X):
        # v1.1: the first rows are repeated header rows ('filename', 'experiment', ...)
        hdr = [i for i in range(len(meta)) if str(meta[i][0]) == "filename"]
        meta = pd.DataFrame([meta[i] for i in range(len(meta)) if i not in hdr], columns=list(meta[hdr[0]]))
        assert len(meta) == len(X), (len(meta), len(X))
        meta["tissue"] = meta.experiment.str.extract(r"(Tissue-[A-Za-z_]+)")[0]
    idx = np.arange(len(X))
    if n and n < len(X):
        idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
        idx.sort()
    if isinstance(meta, pd.DataFrame):
        meta = meta.iloc[idx].reset_index(drop=True)
    elif meta is not None:
        meta = meta[idx]
    return X[idx], y[idx], meta, idx


def norm(ch: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(ch, [1, 99.8])
    return np.clip((ch.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100, help="images to score (0 = all)")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--diameter", type=float, default=0.0)
    args = ap.parse_args()

    X, y, meta, idx = load_split(args.split, args.n or None)
    if isinstance(meta, pd.DataFrame):
        print("tissues:", meta.tissue.value_counts().to_dict())
    if args.inspect:
        print("y whole-cell labels per image (first 5):", [int(len(np.unique(y[i, ..., 0])) - 1) for i in range(min(5, len(y)))])
        return

    import torch
    from cellpose import models

    model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model="cpsam")
    rows = []
    t0 = time.time()
    for k in range(len(X)):
        nuc, mem = X[k, ..., 0], X[k, ..., 1]
        inp = np.stack([norm(mem), norm(nuc), np.zeros_like(nuc, dtype=np.float32)], -1)
        masks, _, _ = model.eval(inp, diameter=args.diameter or None, channel_axis=-1, flow_threshold=0.4, cellprob_threshold=0.0)
        m = match_f1(y[k, ..., 0].astype(np.int64), masks.astype(np.int64))
        row = dict(index=int(idx[k]), n_gt=int(len(np.unique(y[k, ..., 0])) - 1), n_pred=int(masks.max()), **m)
        if isinstance(meta, pd.DataFrame):
            row["tissue"] = meta.tissue.iloc[k]; row["specimen"] = meta.specimen.iloc[k]
            row["pixel_size"] = meta.pixel_size.iloc[k]; row["experiment"] = str(meta.experiment.iloc[k]).split("/")[-1]
        rows.append(row)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(X)}  running mean F1 {np.mean([r['f1'] for r in rows]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"tissuenet_cellpose_{args.split}.csv"
    df.to_csv(out, index=False)
    print(f"\nTissueNet {args.split} n={len(df)}: Cellpose cpsam whole-cell F1@0.5 = {df.f1.mean():.3f} "
          f"(P {df.precision.mean():.3f} R {df.recall.mean():.3f})  wrote {out}")


if __name__ == "__main__":
    main()
