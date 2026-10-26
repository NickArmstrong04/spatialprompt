"""Phenotype benchmark on the Xenium ROIs: does a query return the same cells on our masks as on 10x's?

Reference per (ROI, query) = the gate applied to 10x's own masks with 10x's own counts matrix
(cell_feature_matrix, aligned by label_id). Candidates = the same gate on our masks (SAM, SAM + Muse
QC, Cellpose) with our quantifier. A candidate cell is matched to the 10x cell under its centroid.

  TP = kept candidate whose matched 10x cell is kept     FP = kept candidate with no kept 10x match
  FN = kept 10x cell with no kept candidate matched to it
Precision / recall / F1 per (roi, query, mask set); "10x-requant" = 10x masks with our quantifier,
which isolates quantification noise from segmentation. Query parsing uses the rules backend by
default (no API calls); --backend muse uses Muse Spark (cached for the demo queries).

Writes results/phenotype_eval.csv.  Usage: <spatial python> -X utf8 bench/phenotype_eval.py [--only roi01] [--sets sam sam+muse cellpose]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "bench"))
from spatialprompt.cli import MASK_FILES  # noqa: E402
from spatialprompt.gate import gate  # noqa: E402
from spatialprompt.quantify import quantify_transcripts  # noqa: E402
from spatialprompt.query import parse_query  # noqa: E402
from quantify_check import load_10x_counts  # noqa: E402

QUERIES = [
    "CD8+ T cells",
    "CD8+ T cells at the tumor boundary",
    "CD8+ T cells inside the tumor",
    "tumor cells",
    "HER2+ tumor cells",
    "proliferating tumor cells",
    "macrophages",
    "B cells",
    "endothelial cells",
    "regulatory T cells near the tumor",
]


def kept_ids(label: np.ndarray, counts, genes, table: pd.DataFrame, q: dict, px: float) -> np.ndarray:
    keep, _ = gate(table, counts, genes, q, label, px)
    return table.label.to_numpy()[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sets", nargs="*", default=["sam", "sam+muse", "cellpose", "10x-requant"])
    ap.add_argument("--queries", nargs="*", default=QUERIES)
    ap.add_argument("--backend", choices=["rules", "muse", "auto"], default="auto")
    a = ap.parse_args()
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / "transcripts.parquet").exists())
    if a.only:
        rois = [p for p in rois if p.name in a.only]
    rows = []
    for roi in rois:
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        tr = pd.read_parquet(roi / "transcripts.parquet", columns=["x_px", "y_px", "feature_name", "qv", "is_gene"])
        m10, genes10, cells10 = load_10x_counts(roi)
        order = np.argsort(genes10); genes = genes10[order]; m10 = m10[:, order].tocsr()
        gt = tifffile.imread(roi / "cells_10x.tif").astype(np.int64)
        # reference: 10x masks + 10x's own matrix (rows aligned to the label ids present in the raster)
        lab_to_row = pd.Series(np.arange(len(cells10)), index=cells10.label_id.to_numpy())
        ids = np.unique(gt); ids = ids[ids > 0]; ids = ids[np.isin(ids, lab_to_row.index)]
        ref_counts = m10[lab_to_row[ids].to_numpy()]
        cents = np.array(ndi.center_of_mass(gt > 0, gt, index=ids), dtype=float)
        ref_table = pd.DataFrame(dict(label=ids, x_px=cents[:, 1], y_px=cents[:, 0]))
        queries = {t: parse_query(t, list(genes), backend=a.backend) for t in a.queries}
        # candidates
        cand = {}
        for s in a.sets:
            f = roi / (MASK_FILES["10x"] if s == "10x-requant" else MASK_FILES[s])
            if not f.exists():
                print(f"{roi.name}/{s}: missing {f.name}"); continue
            lab = tifffile.imread(f).astype(np.int64)
            counts, g, table = quantify_transcripts(lab, tr, px, genes=genes)
            # 10x cell under each candidate centroid
            yi = np.clip(table.y_px.round().astype(int), 0, gt.shape[0] - 1); xi = np.clip(table.x_px.round().astype(int), 0, gt.shape[1] - 1)
            table["gt_label"] = gt[yi, xi]
            cand[s] = (lab, counts, g, table)
        for t, q in queries.items():
            ref_keep = set(kept_ids(gt, ref_counts, genes, ref_table, q, px).tolist())
            for s, (lab, counts, g, table) in cand.items():
                ck = kept_ids(lab, counts, g, table, q, px)
                matched_gt = table.set_index("label").loc[ck, "gt_label"].to_numpy()
                tp = int(np.isin(matched_gt, list(ref_keep)).sum()); fp = int(len(ck) - tp)
                fn = int(len(ref_keep - set(matched_gt.tolist())))
                p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f1 = 2 * p * r / max(p + r, 1e-9)
                rows.append(dict(roi=roi.name, query=t, set=s, n_ref=len(ref_keep), n_cand=int(len(ck)), tp=tp, fp=fp, fn=fn,
                                 precision=p, recall=r, f1=f1, backend=q.get("_backend")))
        d = pd.DataFrame([r for r in rows if r["roi"] == roi.name])
        print(f"{roi.name}: " + " | ".join(f"{s}: F1 {g.f1.mean():.2f}" for s, g in d.groupby("set")), flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True); df.to_csv(RESULTS / "phenotype_eval.csv", index=False)
    piv = df.pivot_table(index="query", columns="set", values="f1", aggfunc="mean").round(3)
    nref = df.groupby("query").n_ref.mean().round(0).astype(int)
    piv.insert(0, "n_10x", nref)
    pd.set_option("display.width", 200)
    print("\nF1 per query (mean over ROIs; n_10x = reference cells per ROI):\n" + piv.to_string())
    print("\nmean over queries:\n" + df.groupby("set")[["precision", "recall", "f1"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
