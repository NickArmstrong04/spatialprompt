"""Quantifier: transcripts (Xenium) or channel intensities (CODEX) per mask.

Xenium: a transcript belongs to the cell whose label covers its (x_px, y_px) pixel; this is
exactly 10x's rule (point-in-polygon on the cell boundary) applied to a rasterised label
image. Following 10x's cell_feature_matrix, only transcripts with qv >= 20 and is_gene
(predesigned/custom gene codewords, not controls) are counted.

    counts, genes, table = quantify_transcripts(label, transcripts, px_per_um)
    counts: scipy.sparse CSR (n_cells x n_genes), row i = label id table.label[i]
    table : DataFrame(label, x_px, y_px, x_um, y_um, area_px, area_um2, n_transcripts)
    marker_table(counts, genes, ["CD8A", "CD3E"]) -> DataFrame of those genes per cell

CODEX: mean intensity per channel inside each mask (intensity_table).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import sparse


def cell_table(label: np.ndarray, pixel_size_um: float) -> pd.DataFrame:
    """One row per label id > 0: centroid and area in px and um."""
    lab = np.asarray(label)
    ids = np.unique(lab); ids = ids[ids > 0]
    if len(ids) == 0:
        return pd.DataFrame(columns=["label", "x_px", "y_px", "x_um", "y_um", "area_px", "area_um2"])
    areas = np.bincount(lab.ravel())[ids]
    cents = np.array(ndi.center_of_mass(lab > 0, lab, index=ids), dtype=np.float64)
    return pd.DataFrame(dict(label=ids.astype(np.int64), x_px=cents[:, 1], y_px=cents[:, 0],
                             x_um=cents[:, 1] * pixel_size_um, y_um=cents[:, 0] * pixel_size_um,
                             area_px=areas.astype(np.int64), area_um2=areas * pixel_size_um ** 2))


def quantify_transcripts(label: np.ndarray, transcripts: pd.DataFrame, pixel_size_um: float,
                         genes: np.ndarray | list[str] | None = None, min_qv: float = 20.0,
                         genes_only: bool = True) -> tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    """Assign transcripts to masks by their pixel coordinate.

    transcripts needs columns x_px, y_px, feature_name, and (if filtering) qv, is_gene.
    `genes` fixes the column order (e.g. the panel); default = sorted names present."""
    lab = np.asarray(label)
    H, W = lab.shape
    t = transcripts
    keep = np.ones(len(t), dtype=bool)
    if min_qv is not None and "qv" in t:
        keep &= t["qv"].to_numpy() >= min_qv
    if genes_only and "is_gene" in t:
        keep &= t["is_gene"].to_numpy().astype(bool)
    x = t["x_px"].to_numpy(); y = t["y_px"].to_numpy()
    xi = np.floor(x).astype(np.int64); yi = np.floor(y).astype(np.int64)
    keep &= (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi, yi = xi[keep], yi[keep]
    cell = lab[yi, xi].astype(np.int64)
    names = t["feature_name"].to_numpy()[keep]
    if genes is None:
        genes = np.unique(names)
    genes = np.asarray(genes).astype(str)
    gi = np.searchsorted(genes, names)
    valid = (gi < len(genes)); valid[valid] &= genes[gi[valid]] == names[valid]
    inside = (cell > 0) & valid
    table = cell_table(lab, pixel_size_um)
    row_of = np.full(int(lab.max()) + 1, -1, dtype=np.int64); row_of[table["label"].to_numpy()] = np.arange(len(table))
    rows = row_of[cell[inside]]; cols = gi[inside]
    counts = sparse.coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(table), len(genes))).tocsr()
    counts.sum_duplicates()
    table["n_transcripts"] = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    return counts, genes, table


def marker_table(counts: sparse.spmatrix, genes: np.ndarray, markers: list[str]) -> pd.DataFrame:
    """Dense per-cell columns for a few genes; missing markers raise so typos are loud."""
    genes = np.asarray(genes).astype(str)
    missing = [m for m in markers if m not in set(genes)]
    if missing:
        raise KeyError(f"markers not in panel: {missing}")
    idx = [int(np.where(genes == m)[0][0]) for m in markers]
    return pd.DataFrame(counts[:, idx].toarray(), columns=markers)


def intensity_table(label: np.ndarray, img: np.ndarray, channel_names: list[str]) -> pd.DataFrame:
    """Mean intensity of each channel of img (C, H, W) inside each mask (CODEX / stains)."""
    lab = np.asarray(label)
    ids = np.unique(lab); ids = ids[ids > 0]
    out = {"label": ids.astype(np.int64)}
    for c, name in enumerate(channel_names):
        out[name] = ndi.mean(img[c], labels=lab, index=ids)
    return pd.DataFrame(out)
