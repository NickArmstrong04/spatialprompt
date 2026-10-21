"""SAM 3 cell segmentation: text-prompted seeds -> box-prompted whole-cell masks.

Pipeline (CLAUDE.md "canonical SAM pipeline"):
  1. detect():  tile the RGB composite (384 px, 64 px overlap), run the text prompt "cell"
                on every tile, keep each instance in the tile that owns its centroid, and
                stitch (duplicate suppression + nearest-centroid conflicts). SAM 3's "cell"
                masks are nucleus-sized, so this pass yields the *seeds*.
  2. expand():  for every seed, box-prompt SAM 3's interactive head with the seed's bounding
                box dilated by dilate_um; conflicts go to the nearest seed centroid and every
                seed keeps its own pixels. Output ids == seed ids, so nuclei and cells pair up.
  segment() runs both and returns a Segmentation(cells, nuclei, stats).

prompt_boxes() is the primitive the QC agent uses for repairs: one image, N boxes -> N masks.

Model facts: build_sam3_image_model() loads facebook/sam3 (gated on HF, hf auth login);
num_queries=200 caps instances per forward pass; inputs are resized to 1008 px; everything
runs under bf16 autocast on CUDA.

CLI:  spatialprompt-segment data/roi/roi01/image.tif --out data/roi/roi01  (writes cells_sp.tif, nuclei_sp.tif)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import label as cc_label

from spatialprompt.composite import make_composite, rgb_from_channels


# ----------------------------------------------------------------------------- config

@dataclass
class SegmentConfig:
    prompt: str = "cell"
    tile: int = 384
    overlap: int = 64
    threshold: float = 0.3         # detector confidence threshold
    dilate_um: float = 2.0         # box = seed bbox grown by this on each side
    pixel_size_um: float = 0.2125  # Xenium full-res
    batch: int = 64                # boxes per predict_inst call
    multimask: bool = False        # ask for 3 masks per box and keep the highest predicted IoU
    seed_hint: bool = False        # also feed the dilated seed mask as the dense prompt with each box
    min_area: int = 30             # px; smaller seeds are dropped
    expand: bool = True            # False -> seeds only (nucleus-level segmentation)

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Segmentation:
    cells: np.ndarray                      # uint32 label image, 0 = background
    nuclei: np.ndarray                     # uint32 seeds, same ids as cells
    stats: dict = field(default_factory=dict)

    def save(self, out_dir: Path, prefix: str = "sp") -> None:
        import tifffile
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out_dir / f"cells_{prefix}.tif", self.cells)
        tifffile.imwrite(out_dir / f"nuclei_{prefix}.tif", self.nuclei)
        (out_dir / f"segment_{prefix}.json").write_text(json.dumps(self.stats, indent=2, default=float))


# ----------------------------------------------------------------------------- tiling

def tile_grid(size: int, tile: int, overlap: int) -> list[int]:
    """Start offsets so tiles of tile px with overlap cover [0, size)."""
    if size <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def tile_cores(starts: list[int], tile: int, size: int) -> list[tuple[int, int]]:
    """Ownership interval [c0, c1) per tile: midpoints between neighbouring tile centres,
    so cores partition [0, size) even when the last tile overlaps its neighbour more."""
    centres = [s + tile / 2 for s in starts]
    cores = []
    for i in range(len(starts)):
        c0 = 0 if i == 0 else int(round((centres[i - 1] + centres[i]) / 2))
        c1 = size if i == len(starts) - 1 else int(round((centres[i] + centres[i + 1]) / 2))
        cores.append((c0, c1))
    return cores


def _pad_to(rgb: np.ndarray, tile: int) -> tuple[np.ndarray, int, int]:
    """Images smaller than one tile are zero-padded (TissueNet 256 px, QC crops)."""
    H, W = rgb.shape[:2]
    if H >= tile and W >= tile:
        return rgb, H, W
    out = np.zeros((max(H, tile), max(W, tile), 3), dtype=rgb.dtype)
    out[:H, :W] = rgb
    return out, H, W


# ----------------------------------------------------------------------------- stitching

def stitch(kept: list, H: int, W: int, tile: int, dup_cover: float = 0.5, dup_size: float = 0.5,
           min_area: int = 30) -> tuple[np.ndarray, int]:
    """Resolve overlapping instances (neighbouring tiles, nested detections) into one label image.

    Instances are visited by descending score. A is a duplicate (dropped) if >= dup_cover of its
    pixels already belong to one accepted instance B and A is >= dup_size of B's area. A much
    smaller instance nested inside B (one nucleus inside a two-nucleus blob) is kept and the
    contested pixels go to the nearer centroid. Each survivor is reduced to its largest connected
    component with holes filled; instances under min_area px are dropped.
    kept items are (score, mask[tile, tile] bool, y0, x0)."""
    if not kept:
        return np.zeros((H, W), dtype=np.uint32), 0
    kept = sorted(kept, key=lambda k: -k[0])
    occ = np.zeros((H, W), dtype=np.int32)
    multi = np.zeros((H, W), dtype=bool)
    inst, areas = [], []
    for s, m, y0, x0 in kept:
        a = int(m.sum())
        if a < min_area:
            continue
        sub = occ[y0 : y0 + tile, x0 : x0 + tile]
        ov = sub[m]
        if ov.any():
            counts = np.bincount(ov[ov > 0])
            b = int(np.argmax(counts)); top = counts[b]
            if top / a >= dup_cover and a >= dup_size * areas[b - 1]:
                continue
        k = len(inst) + 1
        yy, xx = np.nonzero(m)
        inst.append((a, yy.mean() + y0, xx.mean() + x0, y0, x0, m)); areas.append(a)
        multi[y0 : y0 + tile, x0 : x0 + tile] |= m & (sub > 0)
        sub[m & (sub == 0)] = k
    n_dedup = len(inst)
    if multi.any():
        bestd = np.full((H, W), np.inf, dtype=np.float32)
        yy_g, xx_g = np.mgrid[0:tile, 0:tile]
        for k, (a, cy, cx, y0, x0, m) in enumerate(inst, start=1):
            win = (slice(y0, y0 + tile), slice(x0, x0 + tile))
            c = m & multi[win]
            if not c.any():
                continue
            d = (yy_g + y0 - cy) ** 2 + (xx_g + x0 - cx) ** 2
            better = c & (d < bestd[win])
            bestd[win][better] = d[better]
            occ[win][better] = k
    out = np.zeros((H, W), dtype=np.uint32); n = 0
    for k, sl in enumerate(ndi.find_objects(occ), start=1):
        if sl is None:
            continue
        sub = occ[sl] == k
        cc = cc_label(sub)
        if cc.max() > 1:
            sizes = np.bincount(cc.ravel())[1:]; sub = cc == (np.argmax(sizes) + 1)
        sub = ndi.binary_fill_holes(sub)
        if sub.sum() < min_area:
            continue
        n += 1
        tgt = out[sl]; tgt[sub & (tgt == 0)] = n
    return out, n_dedup


# ----------------------------------------------------------------------------- model

class Segmenter:
    """Holds one SAM 3 image model (with the interactive head) and runs both passes."""

    def __init__(self, device: str | None = None, threshold: float = 0.3):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        t0 = time.time()
        self.model = build_sam3_image_model(device=self.device, enable_inst_interactivity=True)
        self.proc = Sam3Processor(self.model, confidence_threshold=threshold)
        self.load_sec = time.time() - t0

    def _autocast(self):
        if self.device == "cuda":
            return self._torch.autocast("cuda", dtype=self._torch.bfloat16)
        return contextlib.nullcontext()

    # -- pass 1: text prompt -> seeds -------------------------------------------------
    def detect(self, rgb: np.ndarray, prompt: str = "cell", tile: int = 384, overlap: int = 64,
               threshold: float | None = None, min_area: int = 30) -> tuple[np.ndarray, dict]:
        """Tiled text-prompted detection. Returns (uint32 label image, stats)."""
        if threshold is not None:
            self.proc.set_confidence_threshold(threshold)
        rgb, H, W = _pad_to(rgb, tile)
        Hp, Wp = rgb.shape[:2]
        ys, xs = tile_grid(Hp, tile, overlap), tile_grid(Wp, tile, overlap)
        ycores, xcores = tile_cores(ys, tile, Hp), tile_cores(xs, tile, Wp)
        kept: list[tuple[float, np.ndarray, int, int]] = []
        n_raw = 0; per_tile = []
        t0 = time.time()
        with self._autocast():
            for y0, (gy0, gy1) in zip(ys, ycores):
                for x0, (gx0, gx1) in zip(xs, xcores):
                    state = self.proc.set_image(Image.fromarray(rgb[y0 : y0 + tile, x0 : x0 + tile]))
                    st = self.proc.set_text_prompt(prompt, state)
                    masks = st["masks"].detach().float().cpu().numpy().reshape(-1, tile, tile) > 0.5
                    scores = st["scores"].detach().float().cpu().numpy().ravel()
                    n_raw += len(masks); per_tile.append(len(masks))
                    cy0, cy1, cx0, cx1 = gy0 - y0, gy1 - y0, gx0 - x0, gx1 - x0
                    for m, s in zip(masks, scores):
                        if not m.any():
                            continue
                        yy, xx = np.nonzero(m)
                        cy, cx = yy.mean(), xx.mean()
                        if cy0 <= cy < cy1 and cx0 <= cx < cx1:
                            kept.append((float(s), m, y0, x0))
        lab, n_dedup = stitch(kept, Hp, Wp, tile, min_area=min_area)
        lab = lab[:H, :W]
        stats = dict(n_tiles=len(ys) * len(xs), n_raw=n_raw, n_kept=len(kept), n_dedup=n_dedup,
                     n_seeds=int(lab.max()), max_per_tile=int(max(per_tile)) if per_tile else 0,
                     tiles_ge190=int(sum(v >= 190 for v in per_tile)), detect_sec=time.time() - t0)
        return lab, stats

    # -- primitive: boxes on one image -------------------------------------------------
    def prompt_boxes(self, rgb: np.ndarray, boxes: np.ndarray, multimask: bool = False,
                     batch: int = 64, state=None, mask_hints: np.ndarray | None = None,
                     hint_logit: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
        """Box-prompt SAM 3's interactive head on one image (any size <= ~1008 px is sensible).

        boxes: (N, 4) xyxy in image pixels. Returns (masks (N, H, W) bool, predicted IoU (N,)).
        With multimask=True the head returns 3 candidates per box and we keep the best-scoring.
        mask_hints: optional (N, H, W) bool masks fed as the dense (low-res 256x256 logit) prompt
        alongside each box, +hint_logit inside / -hint_logit outside.
        Pass state (from self.proc.set_image) to reuse the image embedding."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        H, W = rgb.shape[:2]
        out_m = np.zeros((len(boxes), H, W), dtype=bool); out_s = np.zeros(len(boxes), dtype=np.float32)
        if len(boxes) == 0:
            return out_m, out_s
        lowres = None
        if mask_hints is not None:
            from skimage.transform import resize
            # dense prompt resolution = 4x the image-embedding grid (288 px for SAM 3's 1008/14 = 72 grid)
            size = tuple(self.model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size)
            lowres = np.stack([resize(h.astype(np.float32), size, order=1, anti_aliasing=False) for h in mask_hints])
            lowres = np.where(lowres > 0.5, hint_logit, -hint_logit).astype(np.float32)[:, None]  # (N, 1, 288, 288)
        with self._autocast():
            if state is None:
                state = self.proc.set_image(Image.fromarray(rgb))
            for b0 in range(0, len(boxes), batch):
                bb = boxes[b0 : b0 + batch]
                kw = dict(mask_input=lowres[b0 : b0 + batch]) if lowres is not None else {}
                masks, scores, _ = self.model.predict_inst(state, point_coords=None, point_labels=None,
                                                           box=bb, multimask_output=multimask, **kw)
                masks = np.asarray(masks).reshape(len(bb), -1, H, W) > 0
                scores = np.asarray(scores, dtype=np.float32).reshape(len(bb), -1)
                pick = np.argmax(scores, axis=1) if multimask else np.zeros(len(bb), dtype=int)
                out_m[b0 : b0 + len(bb)] = masks[np.arange(len(bb)), pick]
                out_s[b0 : b0 + len(bb)] = scores[np.arange(len(bb)), pick]
        return out_m, out_s

    # -- pass 2: seeds -> whole cells --------------------------------------------------
    def expand(self, rgb: np.ndarray, seeds: np.ndarray, dilate_um: float = 2.0, pixel_size_um: float = 0.2125,
               tile: int = 384, overlap: int = 64, batch: int = 64, multimask: bool = False,
               seed_hint: bool = False) -> tuple[np.ndarray, dict]:
        """Grow every seed into a whole-cell mask by box-prompting. Output ids == seed ids.
        seed_hint=True also passes each seed's dilated mask as the dense prompt."""
        t0 = time.time()
        seeds = np.asarray(seeds).astype(np.int32)
        rgb, H, W = _pad_to(rgb, tile)
        Hp, Wp = rgb.shape[:2]
        if seeds.shape != (Hp, Wp):
            s = np.zeros((Hp, Wp), dtype=np.int32); s[:H, :W] = seeds; seeds = s
        n = int(seeds.max())
        if n == 0:
            return np.zeros((H, W), dtype=np.uint32), dict(n_prompted=0, mean_pred_iou=0.0, expand_sec=0.0)
        d = int(round(dilate_um / pixel_size_um))
        objs = ndi.find_objects(seeds)
        ids_all = np.arange(1, n + 1)
        cents = np.array(ndi.center_of_mass(seeds > 0, seeds, index=ids_all), dtype=np.float64)
        valid = np.array([o is not None for o in objs])
        ys, xs = tile_grid(Hp, tile, overlap), tile_grid(Wp, tile, overlap)
        ycores, xcores = tile_cores(ys, tile, Hp), tile_cores(xs, tile, Wp)
        claim = np.zeros((Hp, Wp), dtype=np.int32)
        multi = np.zeros((Hp, Wp), dtype=bool)
        n_prompted = 0; iou_sum = 0.0
        for y0, (cy0, cy1) in zip(ys, ycores):
            for x0, (cx0, cx1) in zip(xs, xcores):
                sel = valid & (cents[:, 0] >= cy0) & (cents[:, 0] < cy1) & (cents[:, 1] >= cx0) & (cents[:, 1] < cx1)
                ids = ids_all[sel]
                if len(ids) == 0:
                    continue
                boxes = np.array([[max(objs[i - 1][1].start - d, x0) - x0, max(objs[i - 1][0].start - d, y0) - y0,
                                   min(objs[i - 1][1].stop + d, x0 + tile) - x0, min(objs[i - 1][0].stop + d, y0 + tile) - y0]
                                  for i in ids], dtype=np.float32)
                hints = None
                if seed_hint:
                    sub = seeds[y0 : y0 + tile, x0 : x0 + tile]
                    struct = np.ones((2 * d + 1, 2 * d + 1), dtype=bool)
                    hints = np.stack([ndi.binary_dilation(sub == i, structure=struct) for i in ids])
                masks, scores = self.prompt_boxes(rgb[y0 : y0 + tile, x0 : x0 + tile], boxes, multimask=multimask,
                                                  batch=batch, mask_hints=hints)
                win = (slice(y0, y0 + tile), slice(x0, x0 + tile))
                cw, mw = claim[win], multi[win]
                for m, i in zip(masks, ids):
                    mw |= m & (cw > 0)
                    cw[m & (cw == 0)] = i
                n_prompted += len(ids); iou_sum += float(scores.sum())
        if multi.any():  # contested pixels -> nearest seed centroid (Voronoi of centroids)
            marker = np.zeros((Hp, Wp), dtype=np.int32)
            cy = np.clip(np.nan_to_num(cents[:, 0]).round().astype(int), 0, Hp - 1)
            cx = np.clip(np.nan_to_num(cents[:, 1]).round().astype(int), 0, Wp - 1)
            marker[cy[valid], cx[valid]] = ids_all[valid]
            _, inds = ndi.distance_transform_edt(marker == 0, return_indices=True)
            claim[multi] = marker[inds[0], inds[1]][multi]
        own = (seeds > 0) & (claim == 0)
        claim[own] = seeds[own]
        stats = dict(n_prompted=n_prompted, mean_pred_iou=iou_sum / max(n_prompted, 1), expand_sec=time.time() - t0)
        return claim[:H, :W].astype(np.uint32), stats

    # -- both --------------------------------------------------------------------------
    def segment(self, rgb: np.ndarray, cfg: SegmentConfig | None = None) -> Segmentation:
        cfg = cfg or SegmentConfig()
        seeds, st = self.detect(rgb, cfg.prompt, cfg.tile, cfg.overlap, cfg.threshold, cfg.min_area)
        if not cfg.expand:
            return Segmentation(seeds, seeds, dict(config=asdict(cfg), **st))
        cells, st2 = self.expand(rgb, seeds, cfg.dilate_um, cfg.pixel_size_um, cfg.tile, cfg.overlap, cfg.batch, cfg.multimask,
                                 cfg.seed_hint)
        return Segmentation(cells, seeds, dict(config=asdict(cfg), **st, **st2, n_cells=int(len(np.unique(cells)) - 1)))


# ----------------------------------------------------------------------------- cli

def load_rgb(path: Path, composite: str = "dapi_boundary") -> np.ndarray:
    """A 4-channel Xenium stack (.tif, (4, H, W)), a 2-channel [nuclear, boundary] stack, or an RGB image."""
    path = Path(path)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return np.array(Image.open(path).convert("RGB"))
    import tifffile
    img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[0] == 4:
        return make_composite(img, composite)
    if img.ndim == 3 and img.shape[0] == 2:
        return rgb_from_channels(img[0], img[1])
    if img.ndim == 3 and img.shape[-1] == 3:
        return img.astype(np.uint8)
    if img.ndim == 2:
        return rgb_from_channels(img)
    raise ValueError(f"unsupported image shape {img.shape}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SAM 3 text-prompted cell segmentation")
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: image's dir)")
    ap.add_argument("--prefix", default="sp")
    ap.add_argument("--composite", default="dapi_boundary")
    ap.add_argument("--pixel-size-um", type=float, default=None, help="default: roi.json next to the image, else 0.2125")
    for k, v in asdict(SegmentConfig()).items():
        if k == "pixel_size_um":
            continue
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv)
    px = a.pixel_size_um
    if px is None:
        rj = a.image.parent / "roi.json"
        px = json.loads(rj.read_text())["pixel_size_um"] if rj.exists() else 0.2125
    cfg = SegmentConfig.from_dict({**vars(a), "pixel_size_um": px})
    rgb = load_rgb(a.image, a.composite)
    seg = Segmenter(threshold=cfg.threshold)
    print(f"model loaded in {seg.load_sec:.0f}s; segmenting {a.image} ({rgb.shape[1]}x{rgb.shape[0]})", flush=True)
    res = seg.segment(rgb, cfg)
    res.save(a.out or a.image.parent, a.prefix)
    s = res.stats
    print(f"{s['n_seeds']} seeds in {s['detect_sec']:.0f}s" + (f" -> {s['n_cells']} cells in {s['expand_sec']:.0f}s" if cfg.expand else ""))


if __name__ == "__main__":
    main()
