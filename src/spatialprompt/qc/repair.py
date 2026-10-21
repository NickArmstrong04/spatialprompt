"""Repair actions. Each takes the current label image and returns a new one plus a record.

  add_cell(seg, rgb, cells, box)             missed nucleus -> box-prompt SAM, add a new id on free pixels
  split_cell(seg, rgb, cells, cell, nuclei)  doublet -> one box per nucleus, reassign the cell's pixels
  merge_cells(cells, ids)                    split nucleus -> union of the masks under one id
  drop_cell(cells, cell)                     empty mask -> background

Boxes are evaluated on a crop around the target (crop_px square, default 384 = one SAM tile) so
one repair costs one set_image + one predict_inst. Pixel coordinates are (y, x), boxes xyxy.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _crop_window(cy: float, cx: float, H: int, W: int, crop: int) -> tuple[int, int]:
    y0 = int(np.clip(round(cy - crop / 2), 0, max(H - crop, 0)))
    x0 = int(np.clip(round(cx - crop / 2), 0, max(W - crop, 0)))
    return y0, x0


def add_cell(seg, rgb: np.ndarray, cells: np.ndarray, box_yx: tuple[int, int, int, int], dilate_px: int,
             crop: int = 384, min_area: int = 30, max_overlap: float = 0.5, min_pred_iou: float = 0.0,
             max_area: int | None = None) -> tuple[np.ndarray, dict]:
    """box_yx = (y0, y1, x0, x1) of the missed nucleus. Returns (cells, record)."""
    H, W = cells.shape
    y0b, y1b, x0b, x1b = box_yx
    cy, cx = (y0b + y1b) / 2, (x0b + x1b) / 2
    wy, wx = _crop_window(cy, cx, H, W, crop)
    sub = rgb[wy : wy + crop, wx : wx + crop]
    bb = np.array([[max(x0b - dilate_px, wx) - wx, max(y0b - dilate_px, wy) - wy,
                    min(x1b + dilate_px, wx + sub.shape[1]) - wx, min(y1b + dilate_px, wy + sub.shape[0]) - wy]], dtype=np.float32)
    masks, scores = seg.prompt_boxes(sub, bb)
    m = masks[0]
    rec = dict(action="add", y=cy, x=cx, pred_iou=float(scores[0]), ok=False, reason="")
    if m.sum() < min_area:
        rec["reason"] = "mask too small"; return cells, rec
    if scores[0] < min_pred_iou:
        rec["reason"] = "low predicted IoU"; return cells, rec
    if max_area is not None and m.sum() > max_area:
        rec["reason"] = "mask too large"; return cells, rec
    win = cells[wy : wy + crop, wx : wx + crop]
    taken = (win > 0) & m
    if taken.sum() / m.sum() > max_overlap:
        rec["reason"] = "mostly overlaps an existing cell"; return cells, rec
    new_id = int(cells.max()) + 1
    out = cells.copy()
    ow = out[wy : wy + crop, wx : wx + crop]
    free = m & (win == 0)
    # keep the largest component of the free part
    lab, n = ndi.label(free)
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]; free = lab == (np.argmax(sizes) + 1)
    if free.sum() < min_area:
        rec["reason"] = "free area too small"; return cells, rec
    ow[free] = new_id
    rec.update(ok=True, new_id=new_id, area_px=int(free.sum()))
    return out, rec


def split_cell(seg, rgb: np.ndarray, cells: np.ndarray, cell: int, boxes_yx: list[tuple[int, int, int, int]],
               dilate_px: int, crop: int = 384, min_area: int = 30) -> tuple[np.ndarray, dict]:
    """Doublet: prompt one box per nucleus (boxes_yx = [(y0, y1, x0, x1), ...], first = the cell's own seed),
    reassign only the cell's own pixels. The first box keeps the cell id."""
    H, W = cells.shape
    ys, xs = np.nonzero(cells == cell)
    if len(ys) == 0:
        return cells, dict(action="split", cell=cell, ok=False, reason="cell gone")
    cy, cx = ys.mean(), xs.mean()
    wy, wx = _crop_window(cy, cx, H, W, crop)
    sub = rgb[wy : wy + crop, wx : wx + crop]
    bb = []
    for (y0, y1, x0, x1) in boxes_yx:
        bb.append([max(x0 - dilate_px, wx) - wx, max(y0 - dilate_px, wy) - wy,
                   min(x1 + dilate_px, wx + sub.shape[1]) - wx, min(y1 + dilate_px, wy + sub.shape[0]) - wy])
    masks, scores = seg.prompt_boxes(sub, np.array(bb, dtype=np.float32))
    win = cells[wy : wy + crop, wx : wx + crop]
    own = win == cell
    # assign each of the cell's pixels to the box whose mask claims it; ties/unclaimed -> nearest box centre
    cents = np.array([[(y0 + y1) / 2 - wy, (x0 + x1) / 2 - wx] for (y0, y1, x0, x1) in boxes_yx])
    nuclei = boxes_yx
    yy, xx = np.nonzero(own)
    claim = np.full(len(yy), -1)
    for k, m in enumerate(masks):
        hit = m[yy, xx] & (claim < 0)
        claim[hit] = k
    d = ((yy[:, None] - cents[None, :, 0]) ** 2 + (xx[:, None] - cents[None, :, 1]) ** 2)
    nearest = np.argmin(d, axis=1)
    claim[claim < 0] = nearest[claim < 0]
    parts = [int((claim == k).sum()) for k in range(len(nuclei))]
    if min(parts) < min_area:
        return cells, dict(action="split", cell=cell, ok=False, reason="a part is too small", parts=parts)
    out = cells.copy(); ow = out[wy : wy + crop, wx : wx + crop]
    new_ids = [cell] + [int(cells.max()) + 1 + k for k in range(len(nuclei) - 1)]
    for k, nid in enumerate(new_ids):
        ow[yy[claim == k], xx[claim == k]] = nid
    return out, dict(action="split", cell=cell, ok=True, new_ids=new_ids, parts=parts, pred_iou=float(scores.mean()))


def merge_cells(cells: np.ndarray, ids: list[int]) -> tuple[np.ndarray, dict]:
    keep = int(min(ids))
    out = cells.copy()
    out[np.isin(out, ids)] = keep
    return out, dict(action="merge", cells=list(map(int, ids)), kept=keep, ok=True)


def drop_cell(cells: np.ndarray, cell: int) -> tuple[np.ndarray, dict]:
    out = cells.copy(); out[out == cell] = 0
    return out, dict(action="drop", cell=int(cell), ok=True)
