"""Smoke test for MusePlanner: one Muse call on the first n flags of one ROI, decisions vs 10x truth.

No SAM model is loaded (the planner only needs the flag tables and the images), so this runs in
seconds plus one API call. Use it to settle: does json_schema + images work, is the batch size /
max_tokens right, is "minimal" reasoning enough. Batch PNGs go to results/qc_muse/<roi>/smoke/.

    <spatial python> -X utf8 bench/qc_muse_smoke.py --roi roi01 --n 8 --kinds missed
    ... --render-only          # just write the crops, no API call
    ... --no-schema --no-images --effort minimal --batch 16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "bench"))
from spatialprompt import make_composite, muse  # noqa: E402
from spatialprompt.qc import MusePlanner, dapi_foreground, detect_all  # noqa: E402
from spatialprompt.qc.agent import seed_boxes  # noqa: E402
from qc_truth import flag_truth, truth_sets  # noqa: E402

TAG = "dapi_boundary_cell_t384_c0.3_d2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default="roi01")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--kinds", nargs="+", default=["missed"])
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()

    roi = ROOT / "data" / "roi" / a.roi
    px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
    img = tifffile.imread(roi / "image.tif"); dapi = img[0]; rgb = make_composite(img, "dapi_boundary")
    gt_c = tifffile.imread(roi / "cells_10x.tif").astype(np.int64); gt_n = tifffile.imread(roi / "nuclei_10x.tif").astype(np.int64)
    sam = tifffile.imread(roi / f"cells_sp_{TAG}.tif").astype(np.int64); seeds = tifffile.imread(roi / f"nuclei_sp_{TAG}.tif").astype(np.int64)
    fg = dapi_foreground(dapi, px); fg_low = dapi_foreground(dapi, px, otsu_scale=0.45)
    fl = detect_all(sam, seeds, fg, px, fg_low=fg_low)
    truth = flag_truth(fl, truth_sets(gt_c, gt_n, sam, seeds), gt_c)
    print({k: len(v) for k, v in fl.items()}, "| true:", {k: int(truth[k].sum()) for k in ("missed", "doublet", "empty")})

    save_dir = ROOT / "results" / "qc_muse" / a.roi / "smoke"
    planner = MusePlanner(batch=a.batch, kinds=tuple(a.kinds), min_conf=a.min_conf, reasoning_effort=a.effort,
                          use_schema=not a.no_schema, with_images=not a.no_images, max_flags=a.n, save_dir=save_dir)
    ctx = dict(pixel_size_um=px, round=0, seed_box=seed_boxes(seeds), rgb=rgb, cells=sam, seeds=seeds, dapi=dapi, fg=fg)
    if a.render_only:
        planner.with_images = True
        # render without calling: reuse plan()'s item loop by monkeypatching _ask
        planner._ask = lambda items, b: ({}, dict(batch=b, n=len(items), cost_usd=0.0, cached=True, finish_reason="skipped", raw=None))  # type: ignore[method-assign]
    before = muse.spent(muse.key_fingerprint(muse.api_key())) if muse.api_key() else 0.0
    t0 = time.time()
    acts = planner.plan(fl, ctx)
    sec = time.time() - t0
    df = pd.DataFrame(planner.log)
    if len(df):
        df["true"] = [bool(truth[r.kind].loc[int(r.flag_id.split(":")[1])]) for r in df.itertuples()]
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(df[["flag_id", "true", "decision", "confidence", "accepted", "reason"]].to_string(index=False))
        acc = df[df.accepted]
        print(f"\naccepted {len(acc)}/{len(df)}; accept-precision {acc.true.mean() if len(acc) else float('nan'):.2f}; "
              f"true flags kept {acc.true.sum()}/{df.true.sum()}; detector precision {df.true.mean():.2f}")
    print(f"{len(acts)} actions, {planner.n_calls} calls ({planner.n_cached} cached), ${planner.cost_usd:.4f}, {sec:.1f}s; "
          f"failures {planner.failures}")
    if muse.api_key():
        print(f"ledger delta ${muse.spent(muse.key_fingerprint(muse.api_key())) - before:.4f}")
    print(f"crops: {save_dir}")


if __name__ == "__main__":
    main()
