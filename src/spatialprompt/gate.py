"""Phenotype gate: structured query -> subset of cells.

A query (see query.py) is a plain dict so it can be produced by Muse Spark, written by hand,
or stored in JSON:

    {
      "positive":  [["CD8A", "CD8B"], ["CD3E", "CD3D"]],   # AND of groups; each group = sum of genes >= min_count
      "negative":  [["EPCAM", "CDH1"]],                    # each group's sum must be < neg_max (default 1)
      "min_count": 1,
      "region":    {"name": "tumor", "positive": [["EPCAM","CDH1","ERBB2","ESR1","GATA3","FOXA1"]], "min_count": 3},
      "spatial":   {"relation": "boundary", "distance_um": 20},   # boundary | inside | outside | within | null
      "sam_prompt": "cell",
      "label": "CD8+ T cells at the tumor boundary"
    }

gate(table, counts, genes, query, label_image, pixel_size_um) -> (keep: bool array over table rows, info dict)

Spatial relations are evaluated on the rasterised region: region cells' masks are burned to a
binary image, lightly closed, and distance transforms give each cell's signed distance to the
region edge (negative = inside). "boundary" keeps cells with |d| <= distance_um, "inside" d < 0,
"outside" d > 0, "within" 0 < d <= distance_um.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import sparse


def _group_sum(counts: sparse.spmatrix, genes: np.ndarray, group: list[str]) -> np.ndarray:
    genes = np.asarray(genes).astype(str)
    idx = [int(np.where(genes == g)[0][0]) for g in group if g in set(genes)]
    missing = [g for g in group if g not in set(genes)]
    if missing:
        raise KeyError(f"genes not on panel: {missing}")
    return np.asarray(counts[:, idx].sum(axis=1)).ravel()


def marker_mask(counts: sparse.spmatrix, genes: np.ndarray, positive: list[list[str]] | None,
                negative: list[list[str]] | None = None, min_count: int = 1, neg_max: int = 1) -> np.ndarray:
    n = counts.shape[0]
    keep = np.ones(n, dtype=bool)
    for grp in positive or []:
        keep &= _group_sum(counts, genes, grp) >= min_count
    for grp in negative or []:
        keep &= _group_sum(counts, genes, grp) < neg_max
    return keep


def signed_distance_to_region(label: np.ndarray, region_labels: np.ndarray, table: pd.DataFrame,
                              pixel_size_um: float, close_um: float = 5.0) -> np.ndarray:
    """Signed distance (um) from each cell centroid to the edge of the region formed by region_labels.
    Negative inside. +inf if the region is empty."""
    lab = np.asarray(label)
    if len(region_labels) == 0:
        return np.full(len(table), np.inf)
    inreg = np.isin(lab, region_labels)
    r = max(int(round(close_um / pixel_size_um)), 1)
    inreg = ndi.binary_closing(inreg, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    inreg = ndi.binary_fill_holes(inreg)
    d_out = ndi.distance_transform_edt(~inreg) * pixel_size_um   # >0 outside
    d_in = ndi.distance_transform_edt(inreg) * pixel_size_um     # >0 inside
    yi = np.clip(table["y_px"].round().astype(int).to_numpy(), 0, lab.shape[0] - 1)
    xi = np.clip(table["x_px"].round().astype(int).to_numpy(), 0, lab.shape[1] - 1)
    return np.where(inreg[yi, xi], -d_in[yi, xi], d_out[yi, xi])


def gate(table: pd.DataFrame, counts: sparse.spmatrix, genes: np.ndarray, query: dict,
         label: np.ndarray | None = None, pixel_size_um: float | None = None) -> tuple[np.ndarray, dict]:
    """Apply marker logic then the optional spatial filter. Returns (keep, info)."""
    keep = marker_mask(counts, genes, query.get("positive"), query.get("negative"),
                       int(query.get("min_count", 1)), int(query.get("neg_max", 1)))
    info = dict(n_cells=int(len(table)), n_marker=int(keep.sum()))
    sp = query.get("spatial") or {}
    rel = sp.get("relation")
    reg = query.get("region")
    if rel and reg:
        if label is None or pixel_size_um is None:
            raise ValueError("spatial filter needs label image and pixel size")
        in_region = marker_mask(counts, genes, reg.get("positive"), reg.get("negative"), int(reg.get("min_count", 1)))
        d = signed_distance_to_region(label, table["label"].to_numpy()[in_region], table, pixel_size_um)
        dist = float(sp.get("distance_um", 20))
        if rel == "boundary":
            s = np.abs(d) <= dist
        elif rel == "inside":
            s = d < 0
        elif rel == "outside":
            s = d > 0
        elif rel == "within":
            s = (d > 0) & (d <= dist)
        else:
            raise ValueError(f"unknown spatial relation {rel!r}")
        info.update(region=reg.get("name", "region"), n_region=int(in_region.sum()), relation=rel, distance_um=dist,
                    n_spatial=int(s.sum()))
        table = table.assign(dist_to_region_um=d)
        keep = keep & s
    info["n_keep"] = int(keep.sum())
    info["table"] = table
    return keep, info
