"""Query parser: plain English -> the structured query dict that gate.py consumes.

Two backends:
  * Muse Spark via the Meta Model API, through spatialprompt.muse.chat (disk cache, per-key spend
    cap, key in MODEL_API_KEY read from .env). Used when the key is set and backend != "rules".
  * A rules backend (no network) that covers the demo vocabulary so the pipeline runs end to
    end without the key and so Muse Spark's output has a reference to be checked against.

Both return the same schema (see gate.py docstring). Muse Spark is asked for JSON only and its
answer is validated against the panel: unknown genes are dropped with a warning, an empty
positive list falls back to the rules backend.

    parse_query("CD8+ T cells at the tumor boundary", panel=genes)  -> dict
"""
from __future__ import annotations

import json
import re
import warnings

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
  "spatial": null or {"relation": one of "boundary","inside","outside","within", "distance_um": number}:
     "boundary" = within distance_um of the region edge on either side, "within" = outside the region but within
     distance_um of it ("near", "adjacent"), "inside"/"outside" ignore distance_um. Use 20 unless the query gives a distance,
  "sam_prompt": a short visual noun phrase for the segmenter, almost always "cell".
Use only gene symbols on the panel; never use a marker listed as absent. Prefer canonical markers: CD8 T = CD8A/CD8B
and CD3E/CD3D; tumor (breast) = EPCAM, CDH1, ERBB2, ESR1, GATA3, FOXA1 with min_count 3; fibroblast = DCN, LUM, PDGFRA.
Put a qualifier such as HER2+, FOXP3+, exhausted or proliferating into the positive groups as its own group."""

# canonical cell-type markers; the prompt lists which of these are on / absent from the panel, so the model
# sees the relevant part of a 5,101-gene panel for ~600 tokens instead of the whole list
MARKER_HINTS = """PTPRC CD3E CD3D CD3G CD2 CD5 CD7 CD8A CD8B CD4 FOXP3 IL2RA CTLA4 PDCD1 LAG3 HAVCR2 TIGIT TOX GZMA GZMB GZMK
GZMH PRF1 NKG7 GNLY KLRD1 KLRB1 NCAM1 FCGR3A IL7R CCR7 SELL TCF7 MS4A1 CD19 CD79A CD79B CD22 PAX5 CR2 FCER2 JCHAIN MZB1
SDC1 XBP1 IGKC IGHG1 CD38 CD68 CD163 CD14 CD86 CSF1R MRC1 ITGAM ITGAX LYZ C1QA C1QB APOE MARCO SPP1 CD1C CLEC9A CLEC10A
LAMP3 IRF8 IRF4 CPA3 TPSAB1 KIT MS4A2 S100A8 S100A9 CSF3R FCGR3B CEACAM8 EPCAM CDH1 KRT8 KRT18 KRT19 KRT5 KRT14 KRT17
KRT7 ERBB2 ESR1 PGR GATA3 FOXA1 AR ELF5 TP63 MKI67 TOP2A PCNA CCNB1 PECAM1 VWF CDH5 KDR CLDN5 PLVAP ACKR1 LYVE1 PROX1
COL1A1 COL1A2 COL3A1 DCN LUM PDGFRA PDGFRB FAP POSTN THY1 ACTA2 TAGLN MYH11 DES CNN1 RGS5 MCAM VIM CXCL13 CXCL9 CXCL10
CD274 IDO1 HLA-DRA""".split()

_GROUPS = {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "positive": _GROUPS,
        "negative": _GROUPS,
        "min_count": {"type": "integer"},
        "region": {"type": ["object", "null"],
                   "properties": {"name": {"type": "string"}, "positive": _GROUPS, "min_count": {"type": "integer"}},
                   "required": ["name", "positive", "min_count"]},
        "spatial": {"type": ["object", "null"],
                    "properties": {"relation": {"type": "string", "enum": ["boundary", "inside", "outside", "within"]},
                                   "distance_um": {"type": "number"}},
                    "required": ["relation", "distance_um"]},
        "sam_prompt": {"type": "string"},
    },
    "required": ["label", "positive", "negative", "min_count", "region", "spatial", "sam_prompt"],
    "additionalProperties": False,
}


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

def panel_note(panel: list[str] | None) -> str:
    if not panel:
        return "Panel: unknown; use canonical gene symbols."
    ps = set(panel)
    on = [g for g in MARKER_HINTS if g in ps]
    off = [g for g in MARKER_HINTS if g not in ps]
    return (f"Panel: {len(panel)} genes. Canonical markers ON the panel: {' '.join(on)}.\n"
            f"ABSENT from the panel (never use): {' '.join(off)}.\n"
            "Other gene symbols may be used if you are confident they are on a 5K pan-tissue panel; they are validated.")


def parse_muse(text: str, panel: list[str] | None, model: str | None = None, reasoning_effort: str = "low") -> dict:
    """Muse Spark backend, through spatialprompt.muse.chat (disk cache + spend cap)."""
    from spatialprompt.muse import chat
    msgs = [dict(role="system", content="You translate spatial-biology cell queries into a JSON gate specification.\n"
                 + SCHEMA_DOC + "\n" + panel_note(panel)),
            dict(role="user", content=text)]
    fmt = {"type": "json_schema", "json_schema": {"name": "gate_query", "schema": QUERY_SCHEMA}}
    # max_tokens caps reasoning + JSON together; most queries reason ~1k tokens, but "exhausted CD8 T cells in
    # the stroma" used 1.4k and was cut off before any JSON, so retry once with more room
    for max_tokens in (1500, 4000):
        out = chat(msgs, model=model, reasoning_effort=reasoning_effort, max_tokens=max_tokens, tag="parse_query",
                   response_format=fmt)
        if out["finish_reason"] != "length":
            break
    raw = out["content"] or ""
    try:
        q = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        q = json.loads(m.group(0)) if m else {}
    q.setdefault("label", text); q.setdefault("sam_prompt", "cell"); q.setdefault("min_count", 1)
    q.setdefault("positive", []); q.setdefault("negative", []); q.setdefault("region", None); q.setdefault("spatial", None)
    q["_raw"] = raw; q["_backend"] = out["model"]; q["_cost_usd"] = out["cost_usd"]; q["_cached"] = out["cached"]
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
    from spatialprompt.muse import BudgetExceeded, api_key
    use_muse = backend == "muse" or (backend == "auto" and api_key())
    if use_muse:
        try:
            q = validate(parse_muse(text, panel), panel)
            if q["positive"] or not parse_rules(text)["positive"]:  # empty markers only for a generic "all cells"
                return q
            warnings.warn("Muse Spark returned no usable markers; falling back to rules")
        except Exception as e:  # noqa: BLE001
            if backend == "muse" or isinstance(e, BudgetExceeded):
                raise
            warnings.warn(f"Muse Spark unavailable ({e}); using rules backend")
    q = validate(parse_rules(text), panel)
    q["_backend"] = "rules"
    return q


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["CD8+ T cells at the tumor boundary", "macrophages inside the tumor", "tumor cells", "B cells within 30 um of vessels"]:
        print(t, "->", json.dumps({k: v for k, v in parse_query(t, backend="rules").items() if not k.startswith("_")}))
