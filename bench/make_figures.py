"""Regenerate the result figures in results/figures/ from the results CSVs (used by the progress deck).

Usage: <spatial python> -X utf8 bench/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
DARK, LIGHT, ACC, ACC2, GREY = "#14213D", "#FBFBF8", "#E07A1F", "#2A7FB8", "#8A94A6"
SAM_CFG = "dapi_boundary_cell_t384_c0.3_d2"
TN_CFG = "t256_c0.3_d2"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16, "axes.edgecolor": DARK, "axes.labelcolor": DARK,
                         "xtick.color": DARK, "ytick.color": DARK, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": LIGHT, "axes.facecolor": LIGHT, "savefig.facecolor": LIGHT})


def xenium_f1() -> None:
    sp = pd.read_csv(RESULTS / "sam3_pipeline.csv"); sp = sp[sp.config == SAM_CFG].set_index("roi").sort_index()
    cp = pd.read_csv(RESULTS / "baseline_cellpose.csv").set_index("roi").sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, (col, title) in zip(axes, [("cell_f1", "Whole-cell F1 vs 10x"), ("nuc_f1", "Nucleus F1 vs 10x")]):
        x = np.arange(len(sp)); w = 0.38
        ax.bar(x - w / 2, sp[col], w, color=ACC, label=f"SAM 3 (mean {sp[col].mean():.3f})")
        ax.bar(x + w / 2, cp.loc[sp.index, col], w, color=ACC2, label=f"Cellpose (mean {cp.loc[sp.index, col].mean():.3f})")
        ax.set_xticks(x); ax.set_xticklabels(sp.index); ax.set_ylim(0, 1); ax.set_title(title, fontsize=18, color=DARK, loc="left")
        ax.legend(frameon=False, loc="lower right"); ax.set_ylabel("matched-instance F1 @ IoU 0.5")
    fig.tight_layout(); fig.savefig(OUT / "xenium_f1.png", dpi=150); plt.close(fig)


def tissuenet_f1() -> None:
    s = pd.read_csv(RESULTS / "tissuenet_sam3_test.csv"); s = s[s.config == TN_CFG].set_index("index")
    c = pd.read_csv(RESULTS / "tissuenet_cellpose_test.csv").set_index("index")
    j = s.join(c[["f1"]].rename(columns={"f1": "cp_f1"}))
    g = j.groupby("tissue")[["f1", "cp_f1"]].mean().sort_values("f1", ascending=False)
    g.index = [t.replace("Tissue-", "").replace("_", " ") for t in g.index]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    x = np.arange(len(g)); w = 0.38
    ax.bar(x - w / 2, g.f1, w, color=ACC, label=f"SAM 3 (all {len(j):,} images: {j.f1.mean():.3f})")
    ax.bar(x + w / 2, g.cp_f1, w, color=ACC2, label=f"Cellpose cpsam ({j.cp_f1.mean():.3f})")
    ax.set_xticks(x); ax.set_xticklabels(g.index); ax.set_ylim(0, 1); ax.set_ylabel("whole-cell F1 @ IoU 0.5 vs human labels")
    ax.set_title("TissueNet test set by tissue", fontsize=18, color=DARK, loc="left"); ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "tissuenet_f1.png", dpi=150); plt.close(fig)


def tissuenet_variants() -> None:
    tn = pd.read_csv(RESULTS / "tissuenet_sam3_test.csv"); tn = tn[tn.config != "smoke"].groupby("config")[["f1", "precision", "recall"]].mean()
    names = {"t256_c0.3_d2": "default (256 px, thr 0.3, 2 µm)", "t256_c0.2_d2": "threshold 0.2", "t256_c0.3_d3": "boxes 3 µm",
             "t256_c0.3_d1": "boxes 1 µm", "t128_c0.3_d2": "128 px tiles"}
    tn = tn.loc[[k for k in names if k in tn.index]].rename(index=names)
    cp = pd.read_csv(RESULTS / "tissuenet_cellpose_test.csv").f1.mean()
    fig, ax = plt.subplots(figsize=(14, 4.6))
    y = np.arange(len(tn)); h = 0.26
    ax.barh(y + h, tn.f1, h, color=ACC, label="F1"); ax.barh(y, tn.precision, h, color=ACC2, label="precision"); ax.barh(y - h, tn.recall, h, color=GREY, label="recall")
    ax.set_yticks(y); ax.set_yticklabels(tn.index); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.axvline(cp, color=DARK, ls="--", lw=1.5); ax.text(cp + 0.006, len(tn) - 0.6, f"Cellpose F1 {cp:.3f}", color=DARK, fontsize=14)
    ax.set_title("TissueNet: every variant loses to the default", fontsize=18, color=DARK, loc="left"); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "tissuenet_variants.png", dpi=150); plt.close(fig)


def capture_gate() -> None:
    q = pd.read_csv(RESULTS / "quantify_check.csv").groupby("set")[["frac_assigned", "median_per_cell"]].mean().loc[["10x", "cellpose", "sp"]]
    q.index = ["10x", "Cellpose", "SAM 3"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]; ax.bar(q.index, q.frac_assigned * 100, color=[DARK, ACC2, ACC]); ax.set_ylim(0, 100); ax.set_ylabel("% of transcripts inside a cell mask")
    for i, v in enumerate(q.frac_assigned * 100):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", color=DARK)
    ax.set_title("Transcript capture (mean of 5 ROIs)", fontsize=18, color=DARK, loc="left")
    ax = axes[1]
    # roi01 gate counts (CLAUDE.md "Query parser + phenotype gate"); recompute via bench/gate_check.py when it exists
    gate = pd.DataFrame({"10x": [96, 48, 365], "Cellpose": [95, 44, 358], "SAM 3": [96, 39, 298]},
                        index=["CD8+ T cells", "...at tumor\nboundary (±20 µm)", "tumor cells"])
    x = np.arange(len(gate)); w = 0.26
    for i, (col, colr) in enumerate(zip(gate.columns, [DARK, ACC2, ACC])):
        ax.bar(x + (i - 1) * w, gate[col], w, color=colr, label=col)
        for xi, v in zip(x + (i - 1) * w, gate[col]):
            ax.text(xi, v + 5, str(v), ha="center", fontsize=12, color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(gate.index); ax.legend(frameon=False); ax.set_title("Gate on roi01: cells returned per query", fontsize=18, color=DARK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "capture_gate.png", dpi=150); plt.close(fig)


def image_crops() -> None:
    roi = ROOT / "data" / "roi" / "roi01"
    Image.open(roi / f"overlay_sp_{SAM_CFG}.png").save(OUT / "overlay_sam_roi01.png")
    Image.open(roi / "overlay_cellpose.png").save(OUT / "overlay_cellpose_roi01.png")
    g = RESULTS / "gate" / "roi01_sam_cd8t_tumor_boundary.png"
    if g.exists():
        Image.open(g).resize((1200, 1200), Image.LANCZOS).save(OUT / "gate_roi01.png")
    im = Image.open(ROOT / "data" / "roi" / "roi_overview.png"); im.thumbnail((1400, 1400)); im.save(OUT / "roi_overview.png")
    ov = Image.open(roi / "composite.png"); ov.thumbnail((1200, 1200)); ov.save(OUT / "roi01_composite.png")


if __name__ == "__main__":
    setup(); xenium_f1(); tissuenet_f1(); tissuenet_variants(); capture_gate(); image_crops()
    print("wrote", OUT)
