"""Day-1 smoke test: can SAM 3 segment cells in a Xenium fluorescence composite at all?

Runs the SAM 3 image model (text prompt) on a 1024 px crop of one ROI with a few
candidate noun phrases, converts the instance masks to a label image, scores each prompt
against 10x's masks (matched F1 at IoU 0.5), and writes overlays so we can look.

Needs the gated weights: `hf auth login` once with a token that has access to
facebook/sam3 (the image model; facebook/sam3.1 is the video multiplex checkpoint).

Usage: <spatial python> -X utf8 scripts/sam3_smoke.py [--roi roi01] [--prompts cell nucleus]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from cellpose_baseline import match_f1  # noqa: E402


def instances_to_label(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """(N, H, W) bool masks -> uint32 label image; higher-score masks win overlaps."""
    lab = np.zeros(masks.shape[-2:], dtype=np.uint32)
    order = np.argsort(scores)  # low first, so high scores paint last
    for k, i in enumerate(order, start=1):
        lab[masks[i]] = k
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default="roi01")
    ap.add_argument("--prompts", nargs="*", default=["cell", "cells", "nucleus", "round cell", "blue blob"])
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    roi = ROOT / "data" / "roi" / args.roi
    out = ROOT / "results" / "sam3_smoke" / args.roi
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.array(Image.open(roi / "composite.png"))
    gt = tifffile.imread(roi / "cells_10x.tif")
    y0 = x0 = (rgb.shape[0] - args.crop) // 2
    rgb_c = rgb[y0 : y0 + args.crop, x0 : x0 + args.crop]
    gt_c = gt[y0 : y0 + args.crop, x0 : x0 + args.crop]
    Image.fromarray(rgb_c).save(out / "input.png")

    if torch.cuda.is_available():
        # the reference notebooks run everything under bf16 autocast; without it the
        # ViT MLP hits a bf16 x fp32 dtype mismatch
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    t0 = time.time()
    model = build_sam3_image_model(device="cuda" if torch.cuda.is_available() else "cpu")
    proc = Sam3Processor(model, confidence_threshold=args.threshold)
    print(f"model loaded in {time.time()-t0:.0f}s")
    state = proc.set_image(Image.fromarray(rgb_c))

    results = {}
    for prompt in args.prompts:
        proc.reset_all_prompts(state)
        state = proc.set_image(Image.fromarray(rgb_c), state)
        t0 = time.time()
        st = proc.set_text_prompt(prompt, state)
        dt = time.time() - t0
        masks = st["masks"]; scores = st["scores"]
        masks = masks.detach().float().cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        scores = scores.detach().float().cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
        masks = masks.reshape(-1, *masks.shape[-2:]).astype(bool)
        lab = instances_to_label(masks, scores)
        m = match_f1(gt_c.astype(np.int64), lab.astype(np.int64))
        results[prompt] = dict(n_masks=int(len(masks)), sec=dt, **m)
        print(f"'{prompt}': {len(masks)} masks in {dt:.1f}s | vs 10x ({len(np.unique(gt_c))-1} cells) F1@0.5={m['f1']:.3f} P={m['precision']:.2f} R={m['recall']:.2f}")
        tifffile.imwrite(out / f"label_{prompt.replace(' ', '_')}.tif", lab)
        ov = rgb_c.copy()
        ov[find_boundaries(gt_c, mode="inner")] = [0, 255, 255]
        ov[find_boundaries(lab, mode="inner")] = [255, 255, 0]
        Image.fromarray(ov).save(out / f"overlay_{prompt.replace(' ', '_')}.png")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
