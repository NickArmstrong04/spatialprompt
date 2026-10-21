"""Query parser: plain English -> the structured query dict that gate.py consumes.

Two backends:
  * Muse Spark via the Meta Model API (OpenAI-SDK-compatible, base https://api.meta.ai/v1,
    key in MODEL_API_KEY, read from .env). Used when the key is set and backend != "rules".
  * A rules backend (no network) that covers the demo vocabulary so the pipeline runs end to
    end without the key and so Muse Spark's output has a reference to be checked against.

Both return the same schema (see gate.py docstring). Muse Spark is asked for JSON only and its
answer is validated against the panel: unknown genes are dropped with a warning, an empty
positive list falls back to the rules backend.

    parse_query("CD8+ T cells at the tumor boundary", panel=genes)  -> dict
"""
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

# ----------------------------------------------------------------------------- vocabulary

# cell-type vocabulary for the Xenium Prime 5K breast panel; each entry = AND of OR-groups
CELL_TYPES: dict[str, dict] = {
    "cd8 t cell":   dict(positive=[["CD8A", "CD8B"], ["CD3E", "CD3D"]], negative=[], min_count=1),
    "cd4 t cell":   dict(positive=[["CD4"], ["CD3E", "CD3D"]], negative=[["CD8A", "CD8B"]], min_count=1),
    "t cell":       dict(positive=[["CD3E", "CD3D", "CD2"]], negative=[], min_count=1),
    "b cell":       dict(positive=[["MS4A1", "CD79A", "CD19"]], negative=[], min_count=1),
    "macrophage":   dict(positive=[["CD68", "CD163", "CD14"]], negative=[], min_count=1),
    "tumor cell":   dict(positive=[["EPCAM", "CDH1", "ERBB2", "ESR1", "GATA3", "FOXA1"]], negative=[], min_count=3),
    "endothelial":  dict(positive=[["PECAM1", "VWF", "CDH5"]], negative=[], min_count=1),
    "fibroblast":   dict(positive=[["COL1A1", "COL1A2", "DCN", "LUM"]], negative=[["EPCAM", "CDH1"]], min_count=2),
    "proliferating": dict(positive=[["MKI67", "TOP2A"]], negative=[], min_count=1),
    "cell":         dict(positive=[], negative=[], min_count=0),
}
ALIASES = {
    "cd8+ t cells": "cd8 t cell", "cd8 t cells": "cd8 t cell", "cd8+ t cell": "cd8 t cell", "cytotoxic t cells": "cd8 t cell",
    "cd8+ cells": "cd8 t cell", "cd4+ t cells": "cd4 t cell", "cd4 t cells": "cd4 t cell", "helper t cells": "cd4 t cell",
    "t cells": "t cell", "t-cells": "t cell", "lymphocytes": "t cell", "b cells": "b cell", "b-cells": "b cell",
    "macrophages": "macrophage", "myeloid cells": "macrophage", "tumor cells": "tumor cell", "tumour cells": "tumor cell",
    "cancer cells": "tumor cell", "epithelial cells": "tumor cell", "endothelial cells": "endothelial", "vessels": "endothelial",
    "fibroblasts": "fibroblast", "stromal cells": "fibroblast", "proliferating cells": "proliferating", "dividing cells": "proliferating",
    "all cells": "cell", "cells": "cell",
}
REGIONS: dict[str, dict] = {
    "tumor":  dict(name="tumor", positive=[["EPCAM", "CDH1", "ERBB2", "ESR1", "GATA3", "FOXA1"]], min_count=3),
    "stroma": dict(name="stroma", positive=[["COL1A1", "COL1A2", "DCN", "LUM"]], min_count=2),
    "vessel": dict(name="vessel", positive=[["PECAM1", "VWF", "CDH5"]], min_count=1),
}
REGION_ALIASES = {"tumour": "tumor", "tumor nest": "tumor", "cancer": "tumor", "epithelium": "tumor", "stromal": "stroma",
                  "vessels": "vessel", "blood vessels": "vessel", "vasculature": "vessel"}

SCHEMA_DOC = """Return ONLY a JSON object with these keys:
  "label": the query restated briefly,
  "positive": list of OR-groups (lists of gene symbols); a cell must have >= min_count summed transcripts in EVERY group,
  "negative": list of OR-groups the cell must have < 1 transcript of (may be empty),
  "min_count": integer (1 for immune markers, 3 for multi-gene epithelial signatures),
  "region": null or {"name": str, "positive": [...], "min_count": int} describing the reference tissue region,
  "spatial": null or {"relation": one of "boundary","inside","outside","within", "distance_um": number},
  "sam_prompt": a short visual noun phrase for the segmenter, almost always "cell".
Use only gene symbols from the provided panel. Prefer canonical markers: CD8 T = CD8A/CD8B and CD3E/CD3D;
tumor (breast) = EPCAM, CDH1, ERBB2, ESR1, GATA3, FOXA1 with min_count 3; fibroblast = COL1A1, COL1A2, DCN, LUM."""


# ----------------------------------------------------------------------------- rules backend

def parse_rules(text: str) -> dict:
    t = text.lower().strip()
    q: dict = dict(label=text, positive=[], negative=[], min_count=1, region=None, spatial=None, sam_prompt="cell")
    # cell type: longest alias that occurs
    best = None
    for alias, key in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in t:
            best = key; break
    if best is None:
        for key in sorted(CELL_TYPES, key=len, reverse=True):
            if key in t:
                best = key; break
    if best is None:
        m = re.findall(r"\b([A-Z][A-Z0-9]{1,7})\+", text)  # "CD8A+ CD3E+" style
        if m:
            q["positive"] = [[g] for g in m]
        else:
            best = "cell"
    if best:
        ct = CELL_TYPES[best]
        q.update(positive=[list(g) for g in ct["positive"]], negative=[list(g) for g in ct["negative"]], min_count=ct["min_count"])
    for g in re.findall(r"\b([A-Z][A-Z0-9]{1,7})-(?![A-Za-z])", text):  # "EPCAM-" negatives
        q["negative"].append([g])
    # region
    region = None
    for alias, key in sorted({**REGION_ALIASES, **{k: k for k in REGIONS}}.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", t):
            region = key; break
    if region and best != "tumor cell" or (region and region != "tumor"):
        q["region"] = dict(REGIONS[region])
    elif region and best == "tumor cell" and any(w in t for w in ("boundary", "edge", "margin", "border", "interface", "front")):
        q["region"] = dict(REGIONS[region])
    # spatial relation
    dist = re.search(r"(\d+(?:\.\d+)?)\s*(?:um|µm|microns?|micrometers?)", t)
    d = float(dist.group(1)) if dist else 20.0
    if q["region"]:
        if any(w in t for w in ("boundary", "edge", "margin", "border", "interface", "front")):
            q["spatial"] = dict(relation="boundary", distance_um=d)
        elif any(w in t for w in ("inside", "within the", "in the", "intratumoral", "infiltrating")):
            q["spatial"] = dict(relation="inside", distance_um=d)
        elif any(w in t for w in ("outside", "away from", "distal", "excluded")):
            q["spatial"] = dict(relation="outside", distance_um=d)
        elif any(w in t for w in ("near", "adjacent", "next to", "close to", "within")):
            q["spatial"] = dict(relation="within", distance_um=d)
        else:
            q["spatial"] = dict(relation="within", distance_um=d)
    return q


# ----------------------------------------------------------------------------- Muse Spark backend

def _load_env() -> None:
    p = Path(__file__).resolve().parents[2] / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def muse_client():
    """OpenAI-SDK client pointed at the Meta Model API. Raises if the key is missing."""
    _load_env()
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY not set (put it in .env)")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=os.environ.get("MODEL_API_BASE_URL", "https://api.meta.ai/v1"))


def parse_muse(text: str, panel: list[str] | None, model: str | None = None) -> dict:
    client = muse_client()
    model = model or os.environ.get("MUSE_MODEL", "muse-spark-1.3")
    panel_note = f"Panel genes (subset shown, {len(panel)} total): {', '.join(sorted(panel)[:400])}" if panel else "Panel: unknown."
    msgs = [dict(role="system", content="You translate spatial-biology cell queries into a JSON gate specification.\n" + SCHEMA_DOC + "\n" + panel_note),
            dict(role="user", content=text)]
    r = client.chat.completions.create(model=model, messages=msgs, temperature=0, response_format={"type": "json_object"})
    raw = r.choices[0].message.content
    try:
        q = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        q = json.loads(m.group(0)) if m else {}
    q.setdefault("label", text); q.setdefault("sam_prompt", "cell"); q.setdefault("min_count", 1)
    q.setdefault("negative", []); q.setdefault("region", None); q.setdefault("spatial", None)
    q["_raw"] = raw; q["_backend"] = model
    return q


# ----------------------------------------------------------------------------- public

def validate(q: dict, panel: list[str] | None) -> dict:
    if panel is None:
        return q
    ps = set(panel)
    def clean(groups):
        out = []
        for g in groups or []:
            kept = [x for x in g if x in ps]
            dropped = [x for x in g if x not in ps]
            if dropped:
                warnings.warn(f"query genes not on panel, dropped: {dropped}")
            if kept:
                out.append(kept)
        return out
    q["positive"] = clean(q.get("positive")); q["negative"] = clean(q.get("negative"))
    if q.get("region"):
        q["region"]["positive"] = clean(q["region"].get("positive"))
        if not q["region"]["positive"]:
            q["region"] = None; q["spatial"] = None
    return q


def parse_query(text: str, panel: list[str] | None = None, backend: str = "auto") -> dict:
    """backend: "auto" (Muse Spark if MODEL_API_KEY is set, else rules), "muse", or "rules"."""
    _load_env()
    use_muse = backend == "muse" or (backend == "auto" and (os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")))
    if use_muse:
        try:
            q = validate(parse_muse(text, panel), panel)
            if q["positive"] or "cell" in text.lower():
                return q
            warnings.warn("Muse Spark returned no usable markers; falling back to rules")
        except Exception as e:  # noqa: BLE001
            if backend == "muse":
                raise
            warnings.warn(f"Muse Spark unavailable ({e}); using rules backend")
    q = validate(parse_rules(text), panel)
    q["_backend"] = "rules"
    return q


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["CD8+ T cells at the tumor boundary", "macrophages inside the tumor", "tumor cells", "B cells within 30 um of vessels"]:
        print(t, "->", json.dumps({k: v for k, v in parse_query(t, backend="rules").items() if not k.startswith("_")}))
