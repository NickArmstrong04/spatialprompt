"""End-to-end: one plain-English query on one ROI -> masks, counts, the gated cells, an overlay.

    spatialprompt-run --roi data/roi/roi01 --query "CD8+ T cells at the tumor boundary"
    python -m spatialprompt.cli --roi data/roi/roi01 --query "HER2+ tumor cells" --masks sam+muse

Steps (each timed, all file-based so they can be re-run and inspected):
  1. masks      SAM 3 text "cell" -> box-prompted whole cells (reused from the ROI dir when present;
                --fresh recomputes), optionally after the QC agent (--masks sam+qc | sam+muse); or 10x /
                Cellpose masks for comparison.
  2. quantify   transcripts -> counts per mask (qv >= 20, genes only; same rule as 10x).
  3. parse      query -> marker / region / spatial spec (Muse Spark when MODEL_API_KEY is set and
                --backend auto|muse; cached; rules backend otherwise).
  4. gate       marker logic + spatial relation to the marker-defined region.
  5. write      <out>/query.json, cells.parquet (every cell, marker columns, keep flag), labels_query.tif
                (uint32, kept cells only), masks_query.geojson (polygons in slide um), overlay.png, summary.json.

Muse spend: a cached query is free; a new query costs ~$0.005; --masks sam+muse with no cached QC result
for the ROI costs ~$0.32 (the CLI says so before it runs). MUSE_OFFLINE=1 forbids any new API call.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries

from spatialprompt.composite import make_composite
from spatialprompt.gate import gate, marker_mask, region_mask
from spatialprompt.quantify import marker_table, quantify_transcripts
from spatialprompt.query import parse_query

SAM_TAG = "dapi_boundary_cell_t384_c0.3_d2"
MASK_FILES = {
    "sam": f"cells_sp_{SAM_TAG}.tif",
    "sam+qc": "cells_sp_qc.tif",
    "sam+muse": "cells_sp_qc_muse.tif",
    "10x": "cells_10x.tif",
    "cellpose": "cells_cellpose.tif",
}
MAGENTA, YELLOW, CYAN = (255, 0, 255), (255, 255, 0), (0, 255, 255)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def log(msg: str, t0: float | None = None) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}" + (f"  ({time.time() - t0:.1f}s)" if t0 else ""), flush=True)


# ----------------------------------------------------------------------------- step 1: masks

def resolve_masks(roi: Path, kind: str, fresh: bool, px: float, img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Label image for the requested mask set, computing SAM / QC results that are not on disk."""
    info: dict = dict(masks=kind)
    f = roi / MASK_FILES[kind]
    if f.exists() and not fresh:
        info["source"] = f.name; info["reused"] = True
        return tifffile.imread(f).astype(np.int64), info
    if kind in ("10x", "cellpose"):
        raise FileNotFoundError(f"{f} not found (10x / Cellpose masks come from scripts/extract_rois.py and bench/cellpose_baseline.py)")
    # SAM pass (reused if present, even when a fresh QC pass is requested)
    from spatialprompt.segment import SegmentConfig, Segmenter  # lazy: loads torch + the model
    rgb = make_composite(img, "dapi_boundary")
    seg = Segmenter()
    cells_f, nuclei_f = roi / MASK_FILES["sam"], roi / f"nuclei_sp_{SAM_TAG}.tif"
    if cells_f.exists() and nuclei_f.exists() and not (fresh and kind == "sam"):
        cells = tifffile.imread(cells_f).astype(np.int64); seeds = tifffile.imread(nuclei_f).astype(np.int64)
        info["sam"] = "reused"
    else:
        t = time.time()
        res = seg.segment(rgb, SegmentConfig(pixel_size_um=px))
        cells, seeds = res.cells.astype(np.int64), res.nuclei.astype(np.int64)
        tifffile.imwrite(cells_f, res.cells); tifffile.imwrite(nuclei_f, res.nuclei)
        info["sam"] = dict(stats=res.stats, sec=time.time() - t)
        log(f"SAM 3: {res.stats.get('n_cells')} cells", t)
    if kind == "sam":
        info["source"] = cells_f.name
        return cells, info
    # QC agent
    from spatialprompt.qc import MusePlanner, QCAgent, RulesPlanner
    planner = MusePlanner(batch=16) if kind == "sam+muse" else RulesPlanner()
    if kind == "sam+muse":
        log("QC agent with Muse Spark: ~20 calls, ~$0.32 if the ROI's flags are not cached (MUSE_OFFLINE=1 to forbid)")
    t = time.time()
    cells2, qlog = QCAgent(seg, planner).run(rgb, cells, seeds, img[0], px, rounds=1)
    tifffile.imwrite(f, cells2)
    (roi / ("qc_log_muse.json" if kind == "sam+muse" else "qc_log.json")).write_text(json.dumps(qlog, indent=2, default=float))
    info["qc"] = dict(n_before=qlog["n_before"], n_after=qlog["n_after"], actions=qlog["rounds"][0]["actions"],
                      cost_usd=qlog.get("planner", {}).get("cost_usd", 0.0), sec=time.time() - t)
    info["source"] = f.name
    log(f"QC ({kind}): {qlog['n_before']} -> {qlog['n_after']} cells, {qlog['rounds'][0]['actions']}", t)
    return cells2.astype(np.int64), info


# ----------------------------------------------------------------------------- step 5: outputs

def _jsonable(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return None if pd.isna(v) else v


def polygons_geojson(label: np.ndarray, ids: np.ndarray, table: pd.DataFrame, px: float, x0_px: float, y0_px: float) -> dict:
    """One Polygon per label id (outer contour of its mask), coordinates in slide micrometres."""
    objs = ndi.find_objects(label)
    feats = []
    row_of = dict(zip(table.label.to_numpy(), range(len(table))))
    for i in ids:
        sl = objs[int(i) - 1]
        if sl is None:
            continue
        ys, xs = sl[0].start, sl[1].start
        m = np.pad(label[sl] == i, 1)
        cs = find_contours(m.astype(float), 0.5)
        if not cs:
            continue
        c = max(cs, key=len)
        coords = [[round((xs + col - 1 + x0_px) * px, 3), round((ys + row - 1 + y0_px) * px, 3)] for row, col in c]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        r = table.iloc[row_of[int(i)]]
        props = {k: _jsonable(v) for k, v in r.items() if k not in ("x_px", "y_px")}
        feats.append(dict(type="Feature", id=int(i), properties=props, geometry=dict(type="Polygon", coordinates=[coords])))
    return dict(type="FeatureCollection", features=feats)


def overlay_png(roi: Path, label: np.ndarray, keep_ids: np.ndarray, marker_ids: np.ndarray, region: np.ndarray | None, out: Path) -> None:
    """composite.png + yellow = marker-positive cells not kept, magenta = kept cells, cyan = region edge."""
    base = roi / "composite.png"
    img = np.array(Image.open(base).convert("RGB")) if base.exists() else np.zeros(label.shape + (3,), np.uint8)
    if region is not None:
        img[find_boundaries(region, mode="inner")] = CYAN
    other = np.setdiff1d(marker_ids, keep_ids)
    if len(other):
        img[find_boundaries(np.where(np.isin(label, other), label, 0), mode="inner")] = YELLOW
    if len(keep_ids):
        kb = find_boundaries(np.where(np.isin(label, keep_ids), label, 0), mode="inner")
        kb = kb | np.roll(kb, 1, 0) | np.roll(kb, 1, 1)  # 2 px so the kept cells stand out
        img[kb] = MAGENTA
    Image.fromarray(img).save(out)


# ----------------------------------------------------------------------------- run

def run(roi: Path, query_text: str, masks: str = "sam+muse", fresh: bool = False, backend: str = "auto",
        out: Path | None = None, geojson: bool = True) -> dict:
    t_all = time.time()
    roi = Path(roi)
    meta = json.loads((roi / "roi.json").read_text())
    px = float(meta["pixel_size_um"])
    out = Path(out) if out else roi.parents[2] / "results" / "runs" / f"{roi.name}_{slug(query_text)}_{masks.replace('+', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"ROI {roi.name} ({meta.get('size_px')} px = {meta.get('size_um')} um, {px} um/px) | query: {query_text!r} | masks: {masks}")

    t = time.time(); img = tifffile.imread(roi / "image.tif")
    panel = np.sort(np.load(roi / "counts_10x_genes.npy", allow_pickle=True).astype(str))
    tr = pd.read_parquet(roi / "transcripts.parquet", columns=["x_px", "y_px", "feature_name", "qv", "is_gene"])
    log(f"loaded image {img.shape}, {len(tr):,} transcripts, {len(panel):,}-gene panel", t)

    t = time.time(); label, minfo = resolve_masks(roi, masks, fresh, px, img)
    n_cells = int(len(np.unique(label)) - 1)
    log(f"masks: {n_cells} cells from {minfo['source']}" + (" (reused)" if minfo.get("reused") else ""), t)

    t = time.time(); counts, genes, table = quantify_transcripts(label, tr, px, genes=panel)
    log(f"quantified: {int(table.n_transcripts.sum()):,} transcripts in cells, median {table.n_transcripts.median():.0f}/cell", t)

    t = time.time(); q = parse_query(query_text, list(panel), backend=backend)
    qinfo = {k: v for k, v in q.items() if not k.startswith("_")}
    log(f"parsed by {q.get('_backend')}" + (f" (cached)" if q.get("_cached") else "") + f": positive {q['positive']} negative {q['negative']} "
        f"min_count {q['min_count']} region {(q.get('region') or {}).get('name')} spatial {q.get('spatial')}", t)

    t = time.time(); keep, ginfo = gate(table, counts, genes, q, label, px)
    table = ginfo.pop("table")
    region_ids = ginfo.pop("region_labels", None)
    log(f"gate: {ginfo['n_marker']} marker-positive" + (f", {ginfo['n_spatial']} pass the spatial filter ({ginfo['relation']} {ginfo['region']}, "
        f"{ginfo['distance_um']:g} um; region = {ginfo['n_region']} cells)" if "relation" in ginfo else "") + f" -> {ginfo['n_keep']} cells kept", t)

    # per-cell table with the query's genes as columns
    qgenes = sorted({g for grp in (q.get("positive") or []) + (q.get("negative") or []) + ((q.get("region") or {}).get("positive") or []) for g in grp})
    mt = marker_table(counts, genes, qgenes) if qgenes else pd.DataFrame(index=table.index)
    table = pd.concat([table.reset_index(drop=True), mt.reset_index(drop=True)], axis=1)
    x0, y0 = float(meta.get("x0_px", 0)), float(meta.get("y0_px", 0))  # ROI origin in the morphology image
    table["x_um_slide"] = (table.x_px + x0) * px; table["y_um_slide"] = (table.y_px + y0) * px
    table["marker_pass"] = marker_mask(counts, genes, q.get("positive"), q.get("negative"), int(q.get("min_count", 1)), int(q.get("neg_max", 1)))
    table["keep"] = keep
    keep_ids = table.label.to_numpy()[keep]; marker_ids = table.label.to_numpy()[table.marker_pass.to_numpy()]

    t = time.time()
    table.to_parquet(out / "cells.parquet", index=False)
    tifffile.imwrite(out / "labels_query.tif", np.where(np.isin(label, keep_ids), label, 0).astype(np.uint32))
    (out / "query.json").write_text(json.dumps(dict(text=query_text, backend=q.get("_backend"), cached=q.get("_cached"),
                                                    cost_usd=q.get("_cost_usd", 0.0), query=qinfo), indent=2))
    if geojson:
        (out / "masks_query.geojson").write_text(json.dumps(polygons_geojson(label, keep_ids, table, px, x0, y0)))
    region = region_mask(label, region_ids, px) if region_ids is not None and len(region_ids) else None
    overlay_png(roi, label, keep_ids, marker_ids, region, out / "overlay.png")
    summary = dict(roi=roi.name, query=query_text, masks=masks, mask_info=minfo, n_cells=n_cells, parsed=qinfo,
                   parse_backend=q.get("_backend"), gate={k: v for k, v in ginfo.items()}, n_keep=int(keep.sum()),
                   outputs=sorted(p.name for p in out.iterdir()), sec=time.time() - t_all)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out}", t)
    log(f"done: {int(keep.sum())} '{query_text}' cells of {n_cells} in {time.time() - t_all:.1f}s")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="spatialprompt: plain-English cell query on one ROI -> masks + per-cell counts")
    ap.add_argument("--roi", type=Path, required=True, help="ROI dir with image.tif, transcripts.parquet, roi.json, counts_10x_genes.npy")
    ap.add_argument("--query", required=True)
    ap.add_argument("--masks", choices=list(MASK_FILES), default="sam+muse")
    ap.add_argument("--fresh", action="store_true", help="recompute SAM / QC masks even if present")
    ap.add_argument("--backend", choices=["auto", "muse", "rules"], default="auto")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-geojson", action="store_true")
    a = ap.parse_args(argv)
    run(a.roi, a.query, a.masks, a.fresh, a.backend, a.out, not a.no_geojson)


if __name__ == "__main__":
    main(sys.argv[1:])
