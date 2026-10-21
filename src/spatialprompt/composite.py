"""Marker channels -> the RGB image SAM 3 sees.

Finding from the day-1 sweep (CLAUDE.md): SAM 3 with the prompt "cell" works best on a
two-colour composite, R = membrane/boundary stain, B = nuclear stain, G empty. Adding the
interior (18S) channel to G lowers F1.
"""
from __future__ import annotations

import numpy as np


def norm8(ch: np.ndarray, lo_q: float = 1, hi_q: float = 99.8) -> np.ndarray:
    """Percentile-stretch one channel to uint8."""
    lo, hi = np.percentile(ch, [lo_q, hi_q])
    return (np.clip((ch.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)


def rgb_from_channels(nuclear: np.ndarray, boundary: np.ndarray | None = None,
                      interior: np.ndarray | None = None) -> np.ndarray:
    """General entry point: (H, W) channels -> (H, W, 3) uint8 with R boundary, G interior, B nuclear.
    Omitted channels are zero. With only `nuclear`, returns grayscale."""
    n = norm8(nuclear)
    if boundary is None and interior is None:
        return np.stack([n, n, n], -1)
    zero = np.zeros_like(n)
    return np.stack([norm8(boundary) if boundary is not None else zero,
                     norm8(interior) if interior is not None else zero, n], -1)


def make_composite(img: np.ndarray, kind: str = "dapi_boundary") -> np.ndarray:
    """Xenium morphology stack (4, H, W): DAPI, boundary (ATP1A1/CD45/E-Cad), 18S, aSMA/Vim -> RGB."""
    dapi, bnd, rna, sma = (norm8(img[i]) for i in range(4))
    zero = np.zeros_like(dapi)
    if kind == "rgb":            # R boundary, G 18S, B DAPI (same as composite.png)
        return np.stack([bnd, rna, dapi], -1)
    if kind == "dapi":
        return np.stack([dapi, dapi, dapi], -1)
    if kind == "dapi_boundary":  # canonical
        return np.stack([bnd, zero, dapi], -1)
    if kind == "boundary":
        return np.stack([bnd, bnd, bnd], -1)
    if kind == "all":
        return np.stack([bnd, np.maximum(rna, sma), dapi], -1)
    raise ValueError(f"unknown composite kind {kind!r}")
