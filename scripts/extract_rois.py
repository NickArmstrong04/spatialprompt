"""Cut each ROI in data/roi/rois.json out of the Xenium breast bundle.

Per ROI writes data/roi/<name>/
  image.tif            (4, H, W) uint16, channels DAPI / boundary / 18S / aSMA-Vim, level-0 px
  composite.png        RGB preview: R = boundary stain, G = 18S, B = DAPI
  cells_10x.tif        uint32 label image of 10x's cell masks (0 = background)
  nuclei_10x.tif       uint32 label image of 10x's nucleus masks
  cells_10x.parquet    10x per-cell table restricted to the ROI, with label ids
  counts_10x.npz       sparse gene x cell counts for those cells + gene names
  transcripts.parquet  transcripts inside the ROI with pixel coordinates added
  roi.json             the ROI record

Label ids in cells_10x.tif / nuclei_10x.tif equal 10x's `label_id` so they can be joined.
Usage: conda run -n spatial python -X utf8 scripts/extract_rois.py [--only roi01 roi02]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
import tifffile
import zarr
from PIL import Image
from skimage.draw import polygon as draw_polygon

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "xenium_breast_prime"
ROI_DIR = ROOT / "data" / "roi"


def open_morphology() -> zarr.Array:
    tf = tifffile.TiffFile(RAW / "morphology_focus" / "morphology_focus_0000.ome.tif")
    store = tf.series[0].aszarr(level=0)
    z = zarr.open(store, mode="r")
    if isinstance(z, zarr.Group):  # multiscale group: take the full-resolution array
        z = z["0"]
    return z


def rasterize(bounds: pd.DataFrame, x0_px: int, y0_px: int, size: int, pixel_um: float) -> np.ndarray:
    lab = np.zeros((size, size), dtype=np.uint32)
    for cid, g in bounds.groupby("label_id", sort=False):
        xs = g.vertex_x.values / pixel_um - x0_px
        ys = g.vertex_y.values / pixel_um - y0_px
        rr, cc = draw_polygon(ys, xs, shape=lab.shape)
        lab[rr, cc] = int(cid)
    return lab


def to_uint8(ch: np.ndarray, lo_q: float = 1, hi_q: float = 99.8) -> np.ndarray:
    lo, hi = np.percentile(ch, [lo_q, hi_q])
    return np.clip((ch.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    meta = json.loads((ROI_DIR / "rois.json").read_text())
    px_um = meta["pixel_size_um"]
    rois = [r for r in meta["rois"] if not args.only or r["name"] in args.only]

    img = open_morphology()
    print("morphology zarr", img.shape, img.dtype)
    cells = pd.read_parquet(RAW / "cells.parquet")
    cb = pd.read_parquet(RAW / "cell_boundaries.parquet")
    nb = pd.read_parquet(RAW / "nucleus_boundaries.parquet")
    with h5py.File(RAW / "cell_feature_matrix.h5") as f:
        g = f["matrix"]
        genes = g["features/name"][:].astype(str)
        ftype = g["features/feature_type"][:].astype(str)
        barcodes = g["barcodes"][:].astype(str)
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(int(v) for v in g["shape"][:]))
    keep_genes = ftype == "Gene Expression"
    M = M[keep_genes, :].tocsc()
    genes = genes[keep_genes]
    bc_index = pd.Index(barcodes)
    tp = pq.ParquetFile(RAW / "transcripts.parquet")

    for r in rois:
        out = ROI_DIR / r["name"]
        out.mkdir(parents=True, exist_ok=True)
        x0, y0, n = r["x0_px"], r["y0_px"], r["size_px"]
        x0u, y0u, su = r["x0_um"], r["y0_um"], r["size_um"]
        print(f"\n{r['name']}: px x {x0}:{x0+n} y {y0}:{y0+n}")

        crop = np.asarray(img[:, y0 : y0 + n, x0 : x0 + n])
        tifffile.imwrite(out / "image.tif", crop, photometric="minisblack", metadata={"axes": "CYX", "channels": meta["channels"]})
        rgb = np.stack([to_uint8(crop[1]), to_uint8(crop[2]), to_uint8(crop[0])], -1).astype(np.uint8)
        Image.fromarray(rgb).save(out / "composite.png")
        print("  image", crop.shape, "per-channel max", crop.reshape(4, -1).max(1).tolist())

        sel = cells[(cells.x_centroid >= x0u) & (cells.x_centroid < x0u + su) & (cells.y_centroid >= y0u) & (cells.y_centroid < y0u + su)].copy()
        lab_map = cb.drop_duplicates("cell_id").set_index("cell_id")["label_id"]
        sel["label_id"] = sel.cell_id.map(lab_map)
        sel = sel.dropna(subset=["label_id"])
        sel["label_id"] = sel.label_id.astype(int)
        cb_roi = cb[cb.cell_id.isin(sel.cell_id)]
        nb_roi = nb[nb.cell_id.isin(sel.cell_id)]
        lab_c = rasterize(cb_roi, x0, y0, n, px_um)
        lab_n = rasterize(nb_roi, x0, y0, n, px_um)
        tifffile.imwrite(out / "cells_10x.tif", lab_c)
        tifffile.imwrite(out / "nuclei_10x.tif", lab_n)
        sel.to_parquet(out / "cells_10x.parquet", index=False)
        print(f"  10x cells {len(sel)}  labelled px {int((lab_c>0).sum())}  nuclei labels {len(np.unique(lab_n))-1}")

        col = bc_index.get_indexer(sel.cell_id.values)
        sub = M[:, col].tocsr()
        sp.save_npz(out / "counts_10x.npz", sub)
        np.save(out / "counts_10x_genes.npy", genes)

        tabs = []
        for rg in range(tp.metadata.num_row_groups):
            t = tp.read_row_group(rg)
            m = pc.and_(pc.and_(pc.greater_equal(t["x_location"], x0u), pc.less(t["x_location"], x0u + su)),
                        pc.and_(pc.greater_equal(t["y_location"], y0u), pc.less(t["y_location"], y0u + su)))
            t = t.filter(m)
            if t.num_rows:
                tabs.append(t.to_pandas())
        tr = pd.concat(tabs, ignore_index=True)
        tr["x_px"] = tr.x_location / px_um - x0
        tr["y_px"] = tr.y_location / px_um - y0
        tr.to_parquet(out / "transcripts.parquet", index=False)
        print(f"  transcripts {len(tr):,}  (qv>=20: {(tr.qv>=20).mean()*100:.0f} %)")
        (out / "roi.json").write_text(json.dumps({**r, "pixel_size_um": px_um, "channels": meta["channels"]}, indent=2))
    print("\ndone")


if __name__ == "__main__":
    main()
