"""Pick regions of interest on the Xenium breast slide from 10x's own cell table.

Scores a sliding grid of windows for (a) CD8 T-cell count and (b) a mixed tumor /
non-tumor composition, so that "CD8+ T cells at the tumor boundary" has something to
find. Writes data/roi/rois.json (coordinates in microns and in level-0 pixels) and a
PNG overview of the slide with the chosen windows.

Usage: conda run -n spatial python -X utf8 scripts/pick_rois.py [--n 5] [--size-px 2048]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xenium_breast_prime"
OUT = ROOT / "data" / "roi"
PIXEL_UM = 0.2125


def load_marker_counts(genes: list[str]) -> pd.DataFrame:
    with h5py.File(RAW / "cell_feature_matrix.h5") as f:
        g = f["matrix"]
        names = g["features/name"][:].astype(str)
        shape = tuple(int(v) for v in g["shape"][:])
        m = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        barcodes = g["barcodes"][:].astype(str)
    idx = {n: i for i, n in enumerate(names)}
    cols = {}
    for gname in genes:
        cols[gname] = np.asarray(m[idx[gname], :].todense()).ravel()
    return pd.DataFrame(cols, index=barcodes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--size-px", type=int, default=2048)
    ap.add_argument("--step-um", type=float, default=100.0)
    args = ap.parse_args()

    size_um = args.size_px * PIXEL_UM
    cells = pd.read_parquet(RAW / "cells.parquet").set_index("cell_id")
    genes = ["CD8A", "CD8B", "CD3E", "CD3D", "EPCAM", "CDH1", "ERBB2", "ESR1", "GATA3", "FOXA1", "MS4A1", "PECAM1"]
    mk = load_marker_counts(genes).loc[cells.index]
    df = cells[["x_centroid", "y_centroid", "transcript_counts"]].join(mk)

    # 5K panel: per-gene counts are sparse (CD8A>=1 in 3 % of cells), so gate on >=1 with a co-marker
    df["is_cd8t"] = (df.CD8A + df.CD8B >= 1) & (df.CD3E + df.CD3D >= 1)
    df["is_tumor"] = (df.EPCAM + df.CDH1 + df.ERBB2 + df.ESR1 + df.GATA3 + df.FOXA1) >= 3
    df["is_bcell"] = df.MS4A1 >= 3
    print(f"cells {len(df):,}  CD8 T {df.is_cd8t.sum():,}  tumor {df.is_tumor.sum():,}  B {df.is_bcell.sum():,}")

    # Bin cells on a fine grid (step_um), then score windows of size_um via box sums.
    step = args.step_um
    nx = int(np.ceil(df.x_centroid.max() / step)) + 1
    ny = int(np.ceil(df.y_centroid.max() / step)) + 1
    ix = (df.x_centroid // step).astype(int).values
    iy = (df.y_centroid // step).astype(int).values

    def grid(mask: np.ndarray) -> np.ndarray:
        g = np.zeros((ny, nx), dtype=np.int64)
        np.add.at(g, (iy[mask], ix[mask]), 1)
        return g

    g_all = grid(np.ones(len(df), bool))
    g_cd8 = grid(df.is_cd8t.values)
    g_tum = grid(df.is_tumor.values)
    k = int(round(size_um / step))

    def box(g: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.cumsum(np.pad(g, ((1, 0), (1, 0))), axis=0), axis=1)
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]

    w_all, w_cd8, w_tum = box(g_all), box(g_cd8), box(g_tum)
    tum_frac = np.where(w_all > 0, w_tum / np.maximum(w_all, 1), 0)
    # want: many CD8 T cells, a real tumor/non-tumor mix, and dense tissue
    mix = 1 - np.abs(tum_frac - 0.45) / 0.45
    score = w_cd8 * np.clip(mix, 0, 1) * (w_all > 2000)

    chosen = []
    s = score.copy()
    while len(chosen) < args.n and s.max() > 0:
        j, i = np.unravel_index(np.argmax(s), s.shape)
        x0_um, y0_um = i * step, j * step
        chosen.append(
            dict(
                name=f"roi{len(chosen)+1:02d}",
                x0_um=float(x0_um), y0_um=float(y0_um), size_um=float(size_um),
                x0_px=int(round(x0_um / PIXEL_UM)), y0_px=int(round(y0_um / PIXEL_UM)), size_px=args.size_px,
                n_cells=int(w_all[j, i]), n_cd8t=int(w_cd8[j, i]), tumor_frac=float(tum_frac[j, i]),
            )
        )
        # suppress overlapping windows
        s[max(0, j - k) : j + k, max(0, i - k) : i + k] = 0

    OUT.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset="xenium_breast_prime", pixel_size_um=PIXEL_UM,
                channels=["DAPI", "ATP1A1/CD45/E-Cadherin", "18S", "AlphaSMA/Vimentin"],
                cd8t_rule="CD8A+CD8B>=1 and CD3E+CD3D>=1", tumor_rule="EPCAM+CDH1+ERBB2+ESR1+GATA3+FOXA1>=3", rois=chosen)
    (OUT / "rois.json").write_text(json.dumps(meta, indent=2))
    for r in chosen:
        print(f"{r['name']}: x0={r['x0_um']:.0f} y0={r['y0_um']:.0f} um  cells={r['n_cells']}  CD8T={r['n_cd8t']}  tumor_frac={r['tumor_frac']:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 2, figsize=(14, 9))
    ax[0].imshow(np.log1p(g_all), cmap="gray", origin="upper", extent=[0, nx * step, ny * step, 0])
    ax[0].set_title("all cells (log density)")
    ax[1].imshow(np.log1p(g_cd8), cmap="magma", origin="upper", extent=[0, nx * step, ny * step, 0])
    ax[1].set_title("CD8 T cells")
    for a in ax:
        for r in chosen:
            a.add_patch(Rectangle((r["x0_um"], r["y0_um"]), size_um, size_um, fill=False, ec="cyan", lw=1.5))
            a.text(r["x0_um"], r["y0_um"] - 50, r["name"], color="cyan", fontsize=8)
        a.set_xlabel("x (µm)"); a.set_ylabel("y (µm)")
    fig.tight_layout(); fig.savefig(OUT / "roi_overview.png", dpi=120)
    print("wrote", OUT / "rois.json", "and roi_overview.png")


if __name__ == "__main__":
    main()
