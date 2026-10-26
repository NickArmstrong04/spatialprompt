"""Score the QC detectors and the repair loop on the Xenium ROIs (SAM masks vs 10x).

Truth (bench/qc_truth.py): doublet flag TRUE if the SAM cell holds >= 2 10x nucleus centroids; missed flag TRUE if its
blob centre lies in a 10x cell no SAM cell matches (IoU 0.5) and that holds no SAM seed; empty flag TRUE if the SAM
cell holds no 10x nucleus centroid.

--planner rules (default): RulesPlanner (applies every missed/doublet flag) with add-only / split-only ablations ->
  results/qc_check.csv, data/roi/<roi>/cells_sp_qc.tif, qc_log.json, overlay_sp_qc.png (roi01).
--planner muse: MusePlanner (Muse Spark looks at a crop of every flag and accepts / rejects it) -> one row per
  (roi, planner) in results/qc_check_muse.csv with decision quality per kind (accept-precision = accepted & true /
  accepted; rejection-precision = rejected & false / rejected; retention = accepted & true / true flagged) next to
  cell F1 before / after, calls, $ and seconds; per-flag tables in results/qc_flags/<roi>_<planner>.csv;
  data/roi/<roi>/cells_sp_qc_muse.tif, qc_log_muse.json, overlay_sp_qc_muse.png (roi01). RulesPlanner's
  accept-precision is the detector precision by definition (it accepts everything), so the comparison is honest.
--planner both: both of the above.

Usage: <spatial python> -X utf8 bench/qc_check.py [--only roi01] [--rounds 1] [--planner rules|muse|both]
       [--kinds missed doublet empty] [--max-flags N] [--batch 12] [--min-conf 0.6] [--effort low]
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
from PIL import Image
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT / "data" / "roi"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "bench"))
from spatialprompt import Segmenter, make_composite, match_f1  # noqa: E402
from spatialprompt.qc import MusePlanner, QCAgent, RulesPlanner, dapi_foreground, detect_all  # noqa: E402
from qc_truth import flag_truth, prf, truth_sets  # noqa: E402

TAG = "dapi_boundary_cell_t384_c0.3_d2"
KINDS = ("missed", "doublet", "empty")


def overlay(roi: Path, sam: np.ndarray, cells_new: np.ndarray, gt_c: np.ndarray, name: str) -> None:
    """1024 px centre crop: cyan = 10x, yellow = cells after QC, green = ids added by QC."""
    base = np.array(Image.open(roi / "composite.png")); y0 = x0 = (base.shape[0] - 1024) // 2
    sub = base[y0 : y0 + 1024, x0 : x0 + 1024].copy()
    s_new = cells_new[y0 : y0 + 1024, x0 : x0 + 1024].astype(np.int64)
    sub[find_boundaries(gt_c[y0 : y0 + 1024, x0 : x0 + 1024], mode="inner")] = [0, 255, 255]
    sub[find_boundaries(s_new, mode="inner")] = [255, 255, 0]
    sub[find_boundaries(np.where(s_new > sam.max(), s_new, 0), mode="inner")] = [0, 255, 0]
    Image.fromarray(sub).save(roi / name)


def flag_table(fl: dict, truth: dict, px: float, decisions: pd.DataFrame | None, repairs: list[dict]) -> pd.DataFrame:
    """One row per flag of the judged kinds: features, truth, planner decision, repair outcome."""
    rows = []
    for k in KINDS:
        df = fl[k]
        for idx, r in zip(df.index, df.itertuples(index=False)):
            rows.append(dict(kind=k, flag_id=f"{k}:{idx}", cell=int(getattr(r, "cell", 0) or 0), cy=float(getattr(r, "cy", (r.y0 + r.y1) / 2)),
                             cx=float(getattr(r, "cx", (r.x0 + r.x1) / 2)), area_um2=float(r.area_px * px ** 2),
                             cover=float(getattr(r, "cover", np.nan)), dapi_frac=float(getattr(r, "dapi_frac", np.nan)),
                             true=bool(truth[k].loc[idx])))
    out = pd.DataFrame(rows)
    if decisions is not None and len(decisions):
        d = decisions.set_index("flag_id")[["decision", "confidence", "reason", "accepted", "batch", "cached", "cost_usd"]]
        out = out.join(d, on="flag_id")
    else:  # rules: every missed / doublet flag is accepted, empty flags are not acted on
        out["decision"] = out.kind.map({"missed": "add", "doublet": "split", "empty": "skip"}); out["confidence"] = 1.0
        out["reason"] = "rules"; out["accepted"] = out.decision != "skip"
    rep = {r.get("flag_id"): r for r in repairs if r.get("flag_id")}
    out["applied"] = [bool(rep.get(f, {}).get("ok", False)) for f in out.flag_id]
    out["apply_reason"] = [rep.get(f, {}).get("reason", "") for f in out.flag_id]
    out["pred_iou"] = [rep.get(f, {}).get("pred_iou", np.nan) for f in out.flag_id]
    return out


def decision_metrics(ft: pd.DataFrame, judged: tuple[str, ...]) -> dict:
    m = {}
    for k in judged:
        g = ft[ft.kind == k]
        if not len(g):
            continue
        acc, rej = g[g.accepted.fillna(False).astype(bool)], g[~g.accepted.fillna(False).astype(bool)]
        m[f"{k}_n"] = int(len(g)); m[f"{k}_true"] = int(g.true.sum()); m[f"{k}_accepted"] = int(len(acc))
        m[f"{k}_accept_p"] = float(acc.true.mean()) if len(acc) else float("nan")
        m[f"{k}_reject_p"] = float((~rej.true).mean()) if len(rej) else float("nan")
        m[f"{k}_retention"] = float(acc.true.sum() / g.true.sum()) if g.true.sum() else float("nan")
        m[f"{k}_applied"] = int(g.applied.sum())
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--planner", choices=["rules", "muse", "both"], default="rules")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--max-flags", type=int, default=None)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    do_rules, do_muse = a.planner in ("rules", "both"), a.planner in ("muse", "both")
    seg = Segmenter()
    rois = sorted(p for p in ROI_DIR.iterdir() if p.is_dir() and (p / f"cells_sp_{TAG}.tif").exists())
    if a.only:
        rois = [p for p in rois if p.name in a.only]
    (RESULTS / "qc_flags").mkdir(parents=True, exist_ok=True)
    rows, mrows = [], []
    for roi in rois:
        t0 = time.time()
        px = json.loads((roi / "roi.json").read_text())["pixel_size_um"]
        img = tifffile.imread(roi / "image.tif"); dapi = img[0]; rgb = make_composite(img, "dapi_boundary")
        gt_c = tifffile.imread(roi / "cells_10x.tif").astype(np.int64); gt_n = tifffile.imread(roi / "nuclei_10x.tif").astype(np.int64)
        sam = tifffile.imread(roi / f"cells_sp_{TAG}.tif").astype(np.int64)
        seeds = tifffile.imread(roi / f"nuclei_sp_{TAG}.tif").astype(np.int64)
        fg = dapi_foreground(dapi, px); fg_low = dapi_foreground(dapi, px, otsu_scale=0.45)
        # truth + detectors
        ts = truth_sets(gt_c, gt_n, sam, seeds)
        fl = detect_all(sam, seeds, fg, px, fg_low=fg_low)
        truth = flag_truth(fl, ts, gt_c)
        dp, dr = prf(truth["doublet"], len(ts["true_dbl"])); mp, mr = prf(truth["missed"], len(ts["truly_missed"])); ep, er = prf(truth["empty"], len(ts["true_empty"]))
        base = dict(roi=roi.name, n_sam=int(len(np.unique(sam)) - 1), n_10x=ts["n_10x"], n_unmatched_10x=ts["n_10x"] - ts["n_matched"],
                    true_doublets=len(ts["true_dbl"]), true_missed=len(ts["truly_missed"]), true_empty=len(ts["true_empty"]),
                    flag_doublet=len(fl["doublet"]), flag_missed=len(fl["missed"]), flag_empty=len(fl["empty"]), flag_split=len(fl["split"]),
                    doublet_p=dp, doublet_r=dr, missed_p=mp, missed_r=mr, empty_p=ep, empty_r=er,
                    missed_in_any_10x=float(truth["missed_in_any_10x"].mean()) if len(fl["missed"]) else float("nan"))
        before = match_f1(gt_c, sam)
        print(f"{roi.name}: flags dbl {len(fl['doublet'])} (P {dp:.2f} R {dr:.2f} of {len(ts['true_dbl'])}), missed {len(fl['missed'])} "
              f"(P {mp:.2f} R {mr:.2f} of {len(ts['truly_missed'])}), empty {len(fl['empty'])} (P {ep:.2f} R {er:.2f} of {len(ts['true_empty'])}), "
              f"split {len(fl['split'])} | F1 before {before['f1']:.3f}", flush=True)

        def run(planner, rounds: int):
            cells, log = QCAgent(seg, planner).run(rgb, sam, seeds, dapi, px, rounds=rounds, fg=fg)
            for k in KINDS:  # the agent re-runs detect_all on the same inputs; flag ids must line up
                assert log["rounds"][0]["flags"][f"n_{k}"] == len(fl[k]), f"{k}: {log['rounds'][0]['flags'][f'n_{k}']} vs {len(fl[k])}"
            return cells, log, match_f1(gt_c, cells.astype(np.int64))

        if do_rules:
            row = dict(base)
            variants = {"full": RulesPlanner(), "add_only": RulesPlanner(do_split=False), "split_only": RulesPlanner(do_add=False)}
            logs = {}
            for name, planner in variants.items():
                cells, log, m = run(planner, a.rounds if name == "full" else 1)
                row.update({f"f1_{name}": m["f1"], f"p_{name}": m["precision"], f"r_{name}": m["recall"], f"n_{name}": log["n_after"]})
                logs[name] = (cells, log, m)
            cells_full, log_full, m_full = logs["full"]
            acts = log_full["rounds"][0]["actions"]
            row.update(f1_before=before["f1"], p_before=before["precision"], r_before=before["recall"],
                       **{f"{k}_{mm}": v[mm] for k, v in acts.items() for mm in ("planned", "applied")}, qc_sec=log_full["sec"], sec=time.time() - t0)
            rows.append(row)
            ft = flag_table(fl, truth, px, None, log_full["rounds"][0]["repairs"])
            ft.to_csv(RESULTS / "qc_flags" / f"{roi.name}_rules.csv", index=False)
            mrows.append(dict(roi=roi.name, planner="rules", f1_before=before["f1"], f1_after=m_full["f1"], p_after=m_full["precision"], r_after=m_full["recall"],
                              n_after=log_full["n_after"], **decision_metrics(ft, ("missed", "doublet")), calls=0, cost_usd=0.0, qc_sec=log_full["sec"]))
            print(f"  rules: F1 -> full {row['f1_full']:.3f} (P {row['p_full']:.2f} R {row['r_full']:.2f}) | add-only {row['f1_add_only']:.3f} | "
                  f"split-only {row['f1_split_only']:.3f} | {acts} [{log_full['sec']:.0f}s]", flush=True)
            tifffile.imwrite(roi / "cells_sp_qc.tif", cells_full)
            (roi / "qc_log.json").write_text(json.dumps(log_full, indent=2, default=float))
            if roi.name == "roi01":
                overlay(roi, sam, cells_full, gt_c, "overlay_sp_qc.png")

        if do_muse:
            planner = MusePlanner(batch=a.batch, kinds=tuple(a.kinds), min_conf=a.min_conf, reasoning_effort=a.effort,
                                  max_flags=a.max_flags, save_dir=RESULTS / "qc_muse" / roi.name)
            cells_m, log_m, m_m = run(planner, 1)
            ft = flag_table(fl, truth, px, pd.DataFrame(planner.log), log_m["rounds"][0]["repairs"])
            ft.to_csv(RESULTS / "qc_flags" / f"{roi.name}_muse.csv", index=False)
            dm = decision_metrics(ft[ft.decision.notna()], tuple(a.kinds))
            acts = log_m["rounds"][0]["actions"]
            mrows.append(dict(roi=roi.name, planner="muse", f1_before=before["f1"], f1_after=m_m["f1"], p_after=m_m["precision"], r_after=m_m["recall"],
                              n_after=log_m["n_after"], **dm, calls=planner.n_calls, cached=planner.n_cached, cost_usd=planner.cost_usd,
                              failures=len(planner.failures), budget_exceeded=planner.budget_exceeded, n_judged=len(planner.log), qc_sec=log_m["sec"]))
            print(f"  muse : F1 -> {m_m['f1']:.3f} (P {m_m['precision']:.2f} R {m_m['recall']:.2f}) | {acts} | "
                  + " ".join(f"{k}: acc-P {dm.get(f'{k}_accept_p', float('nan')):.2f} rej-P {dm.get(f'{k}_reject_p', float('nan')):.2f} "
                             f"kept {dm.get(f'{k}_retention', float('nan')):.2f} ({dm.get(f'{k}_accepted', 0)}/{dm.get(f'{k}_n', 0)})" for k in a.kinds if f"{k}_n" in dm)
                  + f" | {planner.n_calls} calls ({planner.n_cached} cached) ${planner.cost_usd:.3f} [{log_m['sec']:.0f}s]", flush=True)
            tifffile.imwrite(roi / "cells_sp_qc_muse.tif", cells_m)
            (roi / "qc_log_muse.json").write_text(json.dumps(log_m, indent=2, default=float))
            if roi.name == "roi01":
                overlay(roi, sam, cells_m, gt_c, "overlay_sp_qc_muse.png")

    RESULTS.mkdir(exist_ok=True)
    if do_rules:
        df = pd.DataFrame(rows); df.to_csv(RESULTS / "qc_check.csv", index=False)
        print(f"\nmean of {len(df)} ROIs: doublet P {df.doublet_p.mean():.2f} R {df.doublet_r.mean():.2f} | missed P {df.missed_p.mean():.2f} R {df.missed_r.mean():.2f} | "
              f"empty P {df.empty_p.mean():.2f} R {df.empty_r.mean():.2f} | cell F1 {df.f1_before.mean():.3f} -> full {df.f1_full.mean():.3f} "
              f"(add-only {df.f1_add_only.mean():.3f}, split-only {df.f1_split_only.mean():.3f}); recall {df.r_before.mean():.3f} -> {df.r_full.mean():.3f}; "
              f"precision {df.p_before.mean():.3f} -> {df.p_full.mean():.3f}")
    if do_muse:
        out = RESULTS / "qc_check_muse.csv"
        dm = pd.DataFrame(mrows)
        if out.exists() and not do_rules:  # keep earlier rules rows next to the new muse rows
            old = pd.read_csv(out); old = old[~((old.planner == "muse") & old.roi.isin(dm.roi))]
            dm = pd.concat([old, dm], ignore_index=True)
        dm.to_csv(out, index=False)
        for pl, g in dm.groupby("planner"):
            cols = [c for c in g.columns if c.endswith(("_accept_p", "_reject_p", "_retention"))]
            print(f"{pl} ({len(g)} ROIs): F1 {g.f1_before.mean():.3f} -> {g.f1_after.mean():.3f} | " + " ".join(f"{c} {g[c].mean():.2f}" for c in cols)
                  + f" | ${g.cost_usd.sum():.3f}")


if __name__ == "__main__":
    main()
