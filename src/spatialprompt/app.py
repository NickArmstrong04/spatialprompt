"""Local web UI for the demo (Gradio): type a query, see Muse's parse, run, inspect the cells and the QC agent.

    python -m spatialprompt.app [--port 7860]      ->  http://127.0.0.1:7860

No logic lives here: runs go through spatialprompt.cli.run, parsing through spatialprompt.query,
spend through spatialprompt.muse. Masks are reused from the ROI dir, so a demo run is ~10 s and $0;
the "offline" switch sets MUSE_OFFLINE so a rehearsal can never bill (uncached queries fall back to
the rules parser). Tabs: Query (run + outputs), QC agent (every Muse decision with its reason and a
crop), Benchmarks (the result CSVs and figures).
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage.segmentation import find_boundaries

from spatialprompt import cli, muse
from spatialprompt.query import parse_query

ROOT = Path(__file__).resolve().parents[2]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"
EXAMPLES = ["CD8+ T cells at the tumor boundary", "HER2+ tumor cells", "B cells", "macrophages", "endothelial cells",
            "proliferating tumor cells", "regulatory T cells near the tumor", "CD8+ T cells inside the tumor"]
CELL_COLS = ["label", "x_um_slide", "y_um_slide", "area_um2", "n_transcripts", "dist_to_region_um"]


# ----------------------------------------------------------------------------- data access

def list_rois() -> list[str]:
    return sorted(p.name for p in ROI_DIR.iterdir() if p.is_dir() and (p / "image.tif").exists()) if ROI_DIR.exists() else []


def mask_choices(roi: str) -> list[str]:
    d = ROI_DIR / roi
    have = [k for k, f in cli.MASK_FILES.items() if (d / f).exists()]
    return have or list(cli.MASK_FILES)


def panel(roi: str) -> list[str]:
    return list(np.sort(np.load(ROI_DIR / roi / "counts_10x_genes.npy", allow_pickle=True).astype(str)))


@functools.lru_cache(maxsize=4)
def roi_images(roi: str, masks: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(composite RGB, SAM cells before QC, cells of the chosen mask set) for the QC crops."""
    d = ROI_DIR / roi
    comp = np.array(Image.open(d / "composite.png").convert("RGB"))
    before = tifffile.imread(d / cli.MASK_FILES["sam"]).astype(np.int64)
    after = tifffile.imread(d / cli.MASK_FILES[masks]).astype(np.int64)
    return comp, before, after


def spend_line() -> str:
    key = muse.api_key()
    if not key:
        return "Muse Spark: no API key set (rules parser only)."
    fp = muse.key_fingerprint(key)
    off = os.environ.get("MUSE_OFFLINE", "") not in ("", "0", "false")
    return (f"Muse Spark spend on this key: **${muse.spent(fp):.3f}** of the ${muse.budget_usd():.2f} cap"
            + (" · **offline mode** (cache only, nothing is billed)" if off else ""))


# ----------------------------------------------------------------------------- query tab

def do_parse(roi: str, query: str, backend: str, offline: bool):
    _set_offline(offline)
    t0 = time.time()
    q = parse_query(query, panel(roi), backend=backend)
    spec = {k: v for k, v in q.items() if not k.startswith("_")}
    how = f"parsed by **{q.get('_backend')}**" + (" (cached, $0)" if q.get("_cached") else (f" (${q.get('_cost_usd', 0):.4f})" if q.get("_cost_usd") else "")) \
        + f" in {time.time() - t0:.1f}s"
    return spec, how, spend_line()


def do_run(roi: str, query: str, masks: str, backend: str, offline: bool):
    _set_offline(offline)
    s = cli.run(ROI_DIR / roi, query, masks, fresh=False, backend=backend)
    out = ROOT / "results" / "runs" / f"{roi}_{cli.slug(query)}_{masks.replace('+', '_')}"
    cells = pd.read_parquet(out / "cells.parquet")
    kept = cells[cells.keep]
    gene_cols = [c for c in cells.columns if c not in CELL_COLS + ["x_px", "y_px", "x_um", "y_um", "area_px", "marker_pass", "keep", "in_region"]]
    show = kept[[c for c in CELL_COLS if c in kept.columns] + gene_cols].round(2).reset_index(drop=True)
    g = s["gate"]
    md = (f"**{s['n_keep']} cells** match *{query}* out of {s['n_cells']} ({masks}; {s['mask_info'].get('source')})  \n"
          f"marker-positive {g['n_marker']}" + (f" · in the spatial filter {g['n_spatial']} ({g['relation']} {g['region']}, {g['distance_um']:g} µm; region {g['n_region']} cells)" if "relation" in g else "")
          + f"  \nparsed by {s['parse_backend']} · {s['sec']:.1f} s · outputs in `{out.relative_to(ROOT)}`")
    files = [str(out / f) for f in ("cells.parquet", "masks_query.geojson", "labels_query.tif", "overlay.png", "query.json", "summary.json") if (out / f).exists()]
    return str(out / "overlay.png"), md, show, files, s["parsed"], spend_line()


def _set_offline(offline: bool) -> None:
    if offline:
        os.environ["MUSE_OFFLINE"] = "1"
    else:
        os.environ.pop("MUSE_OFFLINE", None)


# ----------------------------------------------------------------------------- qc tab

def load_qc(roi: str):
    f = ROI_DIR / roi / "qc_log_muse.json"
    if not f.exists():
        return pd.DataFrame(), "No Muse QC log for this ROI yet (run `bench/qc_check.py --planner muse --only " + roi + "`)."
    log = json.loads(f.read_text())
    pl = log.get("planner", {})
    df = pd.DataFrame(pl.get("decisions", []))
    if len(df):
        applied = {r.get("flag_id"): bool(r.get("ok")) for r in log["rounds"][0].get("repairs", []) if r.get("flag_id")}
        df["applied"] = df.flag_id.map(applied).fillna(False).astype(bool)
        df = df[["kind", "flag_id", "decision", "confidence", "accepted", "applied", "reason", "cell", "cy", "cx"]]
    acts = log["rounds"][0].get("actions", {})
    md = (f"**QC agent on {roi}**: {log['n_before']} → {log['n_after']} cells · {len(df)} flags judged by Muse Spark in "
          f"{pl.get('n_calls', 0)} calls (${pl.get('cost_usd', 0):.3f}) · applied: " + ", ".join(f"{k} {v.get('applied', 0)}/{v.get('planned', 0)}" for k, v in acts.items()))
    bench = RESULTS / "qc_check_muse.csv"
    if bench.exists():
        b = pd.read_csv(bench); b = b[(b.roi == roi) & (b.planner == "muse")]
        if len(b):
            r = b.iloc[-1]
            md += (f"  \ncell F1 vs 10x: {r.f1_before:.3f} → **{r.f1_after:.3f}** · missed-flag accept precision {r.get('missed_accept_p', float('nan')):.2f}, "
                   f"empty {r.get('empty_accept_p', float('nan')):.2f}")
    return df, md


def show_decision(roi: str, df: pd.DataFrame, evt: gr.SelectData, size_um: float = 40.0, scale: int = 3):
    if df is None or not len(df):
        return None, ""
    row = df.iloc[evt.index[0]]
    comp, before, after = roi_images(roi, "sam+muse")
    px = json.loads((ROI_DIR / roi / "roi.json").read_text())["pixel_size_um"]
    size = int(round(size_um / px)); cy, cx = float(row.cy), float(row.cx)
    y0 = int(round(cy - size / 2)); x0 = int(round(cx - size / 2))
    def crop(a):
        out = np.zeros((size, size) + a.shape[2:], a.dtype); H, W = a.shape[:2]
        ys, ye, xs, xe = max(y0, 0), min(y0 + size, H), max(x0, 0), min(x0 + size, W)
        out[ys - y0 : ye - y0, xs - x0 : xe - x0] = a[ys:ye, xs:xe]; return out
    S = size * scale
    img = np.array(Image.fromarray(crop(comp)).resize((S, S), Image.BILINEAR))
    b = np.array(Image.fromarray(crop(before).astype(np.int32)).resize((S, S), Image.NEAREST))
    a = np.array(Image.fromarray(crop(after).astype(np.int32)).resize((S, S), Image.NEAREST))
    img[find_boundaries(b, mode="inner")] = (0, 255, 255)
    img[find_boundaries(a, mode="inner")] = (255, 255, 0)
    c = S // 2; img[c - 1 : c + 2, c - 8 : c + 9] = (255, 0, 255); img[c - 8 : c + 9, c - 1 : c + 2] = (255, 0, 255)
    cap = (f"**{row.kind}** flag `{row.flag_id}` → **{row.decision}** (confidence {row.confidence:.2f}"
           f"{', applied' if row.applied else ', not applied'})  \n*{row.reason}*  \ncyan = masks before QC, yellow = after, magenta cross = flag centre")
    return img, cap


# ----------------------------------------------------------------------------- benchmarks tab

def load_csv(name: str) -> pd.DataFrame:
    f = RESULTS / name
    return pd.read_csv(f).round(3) if f.exists() else pd.DataFrame({"missing": [name]})


def figures() -> list[str]:
    d = RESULTS / "figures"
    return sorted(str(p) for p in d.glob("*.png")) if d.exists() else []


# ----------------------------------------------------------------------------- layout

def build() -> gr.Blocks:
    rois = list_rois()
    roi0 = rois[0] if rois else ""
    offline0 = os.environ.get("MUSE_OFFLINE", "") not in ("", "0", "false")
    with gr.Blocks(title="spatialprompt") as demo:
        gr.Markdown("# spatialprompt\nText-promptable cell segmentation + phenotyping for spatial omics — SAM 3 outlines every cell, "
                    "Muse Spark reads the query and judges the segmentation, the gate returns exactly the cells you asked for.")
        spend = gr.Markdown(spend_line())
        with gr.Tab("Query"):
            with gr.Row():
                roi = gr.Dropdown(rois, value=roi0, label="ROI")
                masks = gr.Radio(mask_choices(roi0) if roi0 else [], value="sam+muse" if roi0 and "sam+muse" in mask_choices(roi0) else None, label="masks")
                backend = gr.Radio(["auto", "muse", "rules"], value="auto", label="query parser")
                offline = gr.Checkbox(value=offline0, label="offline (cache only, never bills)")
            query = gr.Textbox(value=EXAMPLES[0], label="query", placeholder="e.g. CD8+ T cells at the tumor boundary")
            gr.Examples(EXAMPLES, inputs=query)
            with gr.Row():
                parse_btn = gr.Button("Parse"); run_btn = gr.Button("Run", variant="primary")
            how = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=3):
                    overlay = gr.Image(label="overlay — cyan region edge · magenta kept cells · yellow marker-positive, not kept", type="filepath")
                with gr.Column(scale=2):
                    spec = gr.JSON(label="parsed query")
                    summary = gr.Markdown()
                    files = gr.File(label="outputs", file_count="multiple")
            cells = gr.Dataframe(label="kept cells", interactive=False, wrap=True)
            roi.change(lambda r: gr.update(choices=mask_choices(r), value="sam+muse" if "sam+muse" in mask_choices(r) else mask_choices(r)[0]), roi, masks)
            parse_btn.click(do_parse, [roi, query, backend, offline], [spec, how, spend])
            run_btn.click(do_run, [roi, query, masks, backend, offline], [overlay, summary, cells, files, spec, spend])
        with gr.Tab("QC agent"):
            qc_roi = gr.Dropdown(rois, value=roi0, label="ROI")
            qc_md = gr.Markdown()
            with gr.Row():
                qc_df = gr.Dataframe(label="Muse Spark decisions (click a row)", interactive=False, wrap=True, scale=3)
                with gr.Column(scale=2):
                    qc_img = gr.Image(label="flagged cell", type="numpy")
                    qc_cap = gr.Markdown()
            qc_roi.change(load_qc, qc_roi, [qc_df, qc_md])
            qc_df.select(show_decision, [qc_roi, qc_df], [qc_img, qc_cap])
            demo.load(load_qc, qc_roi, [qc_df, qc_md])
        with gr.Tab("Benchmarks"):
            gr.Markdown("Segmentation vs 10x (`qc_check_muse.csv`), phenotype queries vs 10x (`phenotype_eval.csv`), hybrid seeds (`sam3_hybrid.csv`), "
                        "TissueNet vs human labels (`tissuenet_sam3_test.csv` summary in CLAUDE.md).")
            gr.Dataframe(load_csv("qc_check_muse.csv"), label="QC agent: rules vs Muse planner", interactive=False)
            gr.Dataframe(load_csv("phenotype_eval.csv"), label="phenotype benchmark", interactive=False)
            gr.Dataframe(load_csv("sam3_hybrid.csv"), label="hybrid seeds", interactive=False)
            gr.Gallery(figures(), label="figures", columns=3, height=400)
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(argv)
    muse.load_env()
    build().launch(server_name="127.0.0.1", server_port=a.port, share=False, inbrowser=False)


if __name__ == "__main__":
    main()
