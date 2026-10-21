"""Validate the quantifier against 10x: re-count transcripts on 10x's own rasterised masks and
compare with 10x's cell_feature_matrix / cells.parquet, then quantify the SAM and Cellpose masks.

Acceptance (CLAUDE.md day 4): per-cell totals on 10x masks match 10x's transcript_counts within
tolerance. Differences come from rasterising polygons to 0.2125 um pixels (edge transcripts)
and from transcripts 10x assigned to cells whose polygon lies partly outside the ROI.

Writes results/quantify_check.csv (one row per ROI x mask set) and, per ROI,
data/roi/<roi>/cells_<set>.parquet with the per-cell table + marker columns.

Usage: <spatial python> -X utf8 bench/quantify_check.py [--only roi01] [--sets 10x sp cellpose]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from spatialprompt.quantify import marker_table, quantify_transcripts  # noqa: E402

# PTPRC and ACTA2 are not on the Xenium Prime 5K panel (CD45 / aSMA are stains there)
MARKERS = ["CD8A", "CD8B", "CD3E", "CD3D", "EPCAM", "CDH1", "ERBB2", "ESR1", "CD68", "PECAM1"]
MASK_FILES = {"10x": "cells_10x.tif", "sp": "cells_sp_dapi_boundary_cell_t384_c0.3_d2.tif", "cellpose": "cells_cellpose.tif"}


def load_10x_counts(roi: Path) -> tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    m = sparse.load_npz(roi / "counts_10x.npz")  # cells x genes (scipy save_npz)
    genes = np.load(roi / "counts_10x_genes.npy", allow_pickle=True).astype(str)
    cells = pd.read_parquet(roi / "cells_10x.parquet")
    if m.shape == (len(genes), len(cells)) and len(genes) != len(cells):
        m = m.T
    assert m.shape == (len(cells), len(genes)), (m.shape, len(cells), len(genes))
    return m.tocsr(), genes, cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sets", nargs="*", default=list(MASK_FILES))
    a = ap.parse_args()
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / "transcripts.parquet").exists())
    if a.only:
        rois = [p for p in rois if p.name in a.only]
    rows = []
    for roi in rois:
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        tr = pd.read_parquet(roi / "transcripts.parquet", columns=["x_px", "y_px", "feature_name", "qv", "is_gene", "cell_id"])
        m10, genes10, cells10 = load_10x_counts(roi)
        order = np.argsort(genes10); genes = genes10[order]; m10 = m10[:, order]
        for s in a.sets:
            f = roi / MASK_FILES[s]
            if not f.exists():
                print(f"{roi.name}/{s}: missing {f.name}"); continue
            lab = tifffile.imread(f)
            counts, g, table = quantify_transcripts(lab, tr, px, genes=genes)
            table = pd.concat([table, marker_table(counts, g, MARKERS)], axis=1)
            row = dict(roi=roi.name, set=s, n_cells=len(table), n_assigned=int(table.n_transcripts.sum()),
                       frac_assigned=float(table.n_transcripts.sum() / ((tr.qv >= 20) & tr.is_gene).sum()),
                       median_per_cell=float(table.n_transcripts.median()))
            if s == "10x":
                # align by label_id: 10x row i <-> label cells10.label_id[i]
                lab_to_row = pd.Series(np.arange(len(cells10)), index=cells10.label_id.to_numpy())
                present = table.label.isin(lab_to_row.index)
                r10 = lab_to_row[table.label[present]].to_numpy()
                ours = table.n_transcripts[present].to_numpy()
                theirs = cells10.transcript_counts.to_numpy()[r10]
                rel = np.abs(ours - theirs) / np.maximum(theirs, 1)
                ours_m = counts[np.where(present)[0]]; theirs_m = m10[r10]
                diff = (ours_m - theirs_m)
                exact_cells = float((np.asarray(abs(diff).sum(axis=1)).ravel() == 0).mean())
                row.update(n_matched=int(present.sum()), total_ours=int(ours.sum()), total_10x=int(theirs.sum()),
                           cells_within_5pct=float((rel <= 0.05).mean()), cells_within_10pct=float((rel <= 0.10).mean()),
                           cells_exact_matrix=exact_cells, median_abs_rel=float(np.median(rel)),
                           pearson_total=float(np.corrcoef(ours, theirs)[0, 1]),
                           pearson_marker=float(np.corrcoef(np.asarray(ours_m.sum(axis=0)).ravel(), np.asarray(theirs_m.sum(axis=0)).ravel())[0, 1]))
                # where do we lose transcripts? 10x assigns by polygon even outside the ROI crop
                assigned10 = tr.cell_id.isin(set(cells10.cell_id)) & (tr.qv >= 20) & tr.is_gene
                row["frac_10x_assigned_in_roi"] = float(assigned10.sum() / ((tr.qv >= 20) & tr.is_gene).sum())
            rows.append(row)
            table.to_parquet(roi / f"cells_{s}.parquet" if s != "10x" else roi / "cells_10x_requant.parquet", index=False)
            extra = (f" | vs 10x: {row['cells_within_5pct']:.1%} cells within 5%, {row['cells_within_10pct']:.1%} within 10%, "
                     f"{row['cells_exact_matrix']:.1%} exact rows, r={row['pearson_total']:.4f}, totals {row['total_ours']}/{row['total_10x']}") if s == "10x" else ""
            print(f"{roi.name}/{s}: {row['n_cells']} cells, {row['frac_assigned']:.1%} of transcripts assigned, median {row['median_per_cell']:.0f}/cell{extra}", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "quantify_check.csv", index=False)
    print("\n", df.groupby("set")[["n_cells", "frac_assigned", "median_per_cell"]].mean().round(3).to_string())
    if "10x" in a.sets:
        d = df[df.set == "10x"]
        print(f"10x re-quantification: {d.cells_within_5pct.mean():.1%} of cells within 5 %, {d.cells_exact_matrix.mean():.1%} exact, r {d.pearson_total.mean():.4f}")


if __name__ == "__main__":
    main()
