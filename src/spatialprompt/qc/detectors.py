"""QC detectors: find masks that look wrong, using evidence SAM did not use.

Evidence = the DAPI foreground (smoothed DAPI above a fraction of Otsu) compared with the cells
and their seeds (SAM's nucleus-sized text detections). Every SAM cell has exactly one seed by
construction, so "extra nucleus" means DAPI foreground inside a cell that is away from its seed.

    fg = dapi_foreground(dapi, px)
    flags = detect_all(cells, seeds, fg, px)     # dict of DataFrames: doublet, split, empty, missed

  missed  : a DAPI blob (>= min_nuc_um2) with < missed_cover of its area under any cell -> box
  doublet : a cell holding a DAPI blob (>= min_nuc_um2) outside its seed dilated by seed_margin_um -> cell + box
  empty   : a cell whose DAPI-foreground fraction is < empty_frac
  split   : a compact DAPI blob (< split_max_um2) shared by exactly two cells, >= split_frac in each
All detectors are plain functions so the agent can call them as tools and the bench can score them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu


def _norm(ch: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(ch, [1, 99.8])
    return np.clip((ch.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def dapi_foreground(dapi: np.ndarray, pixel_size_um: float, sigma_um: float = 0.3, otsu_scale: float = 0.6) -> np.ndarray:
    """Binary nuclear foreground: smoothed DAPI above otsu_scale x Otsu, holes filled, 1-px opening."""
    d = gaussian(_norm(dapi), sigma=max(sigma_um / pixel_size_um, 0.5), preserve_range=True)
    fg = d > threshold_otsu(d) * otsu_scale
    return ndi.binary_fill_holes(ndi.binary_opening(fg, iterations=1))


def _blobs(mask: np.ndarray, min_px: float) -> tuple[np.ndarray, pd.DataFrame]:
    lab, n = ndi.label(mask)
    if n == 0:
        return lab, pd.DataFrame(columns=["blob", "area_px", "cy", "cx", "y0", "y1", "x0", "x1"])
    areas = np.bincount(lab.ravel())[1:]
    keep = np.where(areas >= min_px)[0] + 1
    objs = ndi.find_objects(lab)
    cents = ndi.center_of_mass(mask, lab, index=keep) if len(keep) else []
    rows = [dict(blob=int(i), area_px=int(areas[i - 1]), cy=float(c[0]), cx=float(c[1]),
                 y0=objs[i - 1][0].start, y1=objs[i - 1][0].stop, x0=objs[i - 1][1].start, x1=objs[i - 1][1].stop)
            for i, c in zip(keep, cents)]
    return lab, pd.DataFrame(rows, columns=["blob", "area_px", "cy", "cx", "y0", "y1", "x0", "x1"])


def detect_all(cells: np.ndarray, seeds: np.ndarray, fg: np.ndarray, pixel_size_um: float,
               min_nuc_um2: float = 10.0, missed_cover: float = 0.25, seed_margin_um: float = 1.0,
               empty_frac: float = 0.05, split_frac: float = 0.3, split_max_um2: float = 60.0,
               fg_low: np.ndarray | None = None, missed_min_um2: float = 8.0) -> dict[str, pd.DataFrame]:
    """fg is the normal DAPI foreground (0.6 x Otsu). fg_low (default: fg) is a more permissive
    foreground used only by the missed detector: the cells SAM misses are dim (median DAPI half
    of a typical nucleus on roi01), so 0.3 x Otsu finds 88 % of them vs 46 % at 0.6."""
    cells = np.asarray(cells).astype(np.int64); seeds = np.asarray(seeds).astype(np.int64)
    min_px = min_nuc_um2 / pixel_size_um ** 2
    covered = cells > 0
    fgl = fg if fg_low is None else fg_low
    # missed: foreground blobs mostly outside every cell
    lab, blobs = _blobs(fgl & ~ndi.binary_dilation(covered, iterations=1), missed_min_um2 / pixel_size_um ** 2)
    rows = []
    for r in blobs.itertuples():
        # coverage of the *whole* DAPI blob this fragment belongs to, so a sliver poking out of a cell is not a miss
        win = (slice(r.y0, r.y1), slice(r.x0, r.x1))
        frag = lab[win] == r.blob
        whole = ndi.binary_dilation(frag, iterations=2) & fgl[win]
        cov = (whole & covered[win]).sum() / max(whole.sum(), 1)
        if cov < missed_cover:
            rows.append(dict(blob=r.blob, area_px=r.area_px, cover=float(cov), cy=r.cy, cx=r.cx, y0=r.y0, y1=r.y1, x0=r.x0, x1=r.x1))
    missed = pd.DataFrame(rows, columns=["blob", "area_px", "cover", "cy", "cx", "y0", "y1", "x0", "x1"])
    # doublet: foreground inside a cell, away from its seed
    m = max(int(round(seed_margin_um / pixel_size_um)), 1)
    seed_zone = ndi.binary_dilation(seeds > 0, iterations=m)
    extra = fg & covered & ~seed_zone
    lab2, blobs2 = _blobs(extra, min_px)
    rows = []
    if len(blobs2):
        owner = cells[blobs2.cy.round().astype(int).clip(0, cells.shape[0] - 1), blobs2.cx.round().astype(int).clip(0, cells.shape[1] - 1)]
        for r, cell in zip(blobs2.itertuples(), owner):
            if cell == 0:
                continue
            win = (slice(r.y0, r.y1), slice(r.x0, r.x1))
            inside = ((lab2[win] == r.blob) & (cells[win] == cell)).sum() / r.area_px
            if inside >= 0.8:
                rows.append(dict(cell=int(cell), blob=r.blob, area_px=r.area_px, cy=r.cy, cx=r.cx, y0=r.y0, y1=r.y1, x0=r.x0, x1=r.x1))
    doublet = pd.DataFrame(rows, columns=["cell", "blob", "area_px", "cy", "cx", "y0", "y1", "x0", "x1"])
    if len(doublet):  # one flag per cell: keep the largest extra blob
        doublet = doublet.sort_values("area_px", ascending=False).drop_duplicates("cell").reset_index(drop=True)
    # bounding box of each cell, for the planner's crops (doublet rows carry the extra blob's box, which is
    # too small to crop on; empty rows have no box at all)
    objs = ndi.find_objects(cells)
    def cell_box(c: int) -> tuple[int, int, int, int]:
        sl = objs[int(c) - 1]
        return (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop) if sl is not None else (0, 0, 0, 0)
    for col, i in zip(("cell_y0", "cell_y1", "cell_x0", "cell_x1"), range(4)):
        doublet[col] = [cell_box(c)[i] for c in doublet.cell] if len(doublet) else pd.Series(dtype=int)
    # empty: little DAPI foreground inside the cell
    ids = np.arange(1, cells.max() + 1)
    area = np.bincount(cells.ravel(), minlength=cells.max() + 1)[1:]
    fg_in = np.bincount(cells[fg].ravel(), minlength=cells.max() + 1)[1:]
    present = area > 0
    frac = np.where(present, fg_in / np.maximum(area, 1), 1.0)
    e = ids[present & (frac < empty_frac)]
    boxes = np.array([cell_box(c) for c in e], dtype=int).reshape(-1, 4)
    empty = pd.DataFrame(dict(cell=e.astype(int), dapi_frac=frac[e - 1], area_px=area[e - 1].astype(int),
                              y0=boxes[:, 0], y1=boxes[:, 1], x0=boxes[:, 2], x1=boxes[:, 3]))
    # split: a compact blob shared by exactly two cells
    lab3, blobs3 = _blobs(fg, min_px)
    rows = []
    max_px = split_max_um2 / pixel_size_um ** 2
    for r in blobs3[blobs3.area_px <= max_px].itertuples():
        win = (slice(r.y0, r.y1), slice(r.x0, r.x1))
        inb = lab3[win] == r.blob
        owners = cells[win][inb]; owners = owners[owners > 0]
        if len(owners) < r.area_px * 0.8:
            continue
        u, n = np.unique(owners, return_counts=True)
        big = u[n / r.area_px >= split_frac]
        if len(big) == 2 and len(u) <= 3:
            rows.append(dict(blob=r.blob, area_px=r.area_px, cells=[int(x) for x in big]))
    split = pd.DataFrame(rows, columns=["blob", "area_px", "cells"])
    return dict(doublet=doublet, split=split, empty=empty, missed=missed)


def summarize(flags: dict[str, pd.DataFrame], n_cells: int) -> dict:
    return dict(n_cells=int(n_cells), **{f"n_{k}": int(len(v)) for k, v in flags.items()})
