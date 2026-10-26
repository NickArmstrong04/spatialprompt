"""MusePlanner: Muse Spark judges the QC detectors' proposals by looking at each flagged cell.

Same `plan(flags, context) -> actions` contract as RulesPlanner. For every flag it renders a two-panel
crop (DAPI grayscale | boundary/DAPI composite, with the current cell outlines and the flag box drawn
on top), batches `batch` flags per Muse call with one line of numbers each, and asks for a JSON list
of decisions: add (missed), split (doublet), drop (empty) or skip, with a confidence and a short
reason. Accepted decisions become the same action dicts RulesPlanner emits, so qc.repair is unchanged.

All calls go through spatialprompt.muse.chat (disk cache + per-key spend cap). Nothing run-specific
goes into the messages, so identical flags on a re-run are free. `self.log` holds one entry per flag
(decision, confidence, reason, cost) and QCAgent copies it into its log for qc_log.json.

    planner = MusePlanner(batch=12, kinds=("missed", "doublet", "empty"))
    cells2, log = QCAgent(seg, planner).run(rgb, cells, seeds, dapi, px)
    log["planner"]["decisions"]
"""
from __future__ import annotations

import base64
import io
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries

from spatialprompt import muse

ALLOWED = {"missed": ("add", "skip"), "doublet": ("split", "skip"), "empty": ("drop", "skip")}
YELLOW, GREEN, CYAN = (255, 255, 0), (0, 255, 0), (0, 255, 255)

SYSTEM_PROMPT = """You are the quality-control reviewer for a cell segmentation of a 10x Xenium image of FFPE human breast cancer.
Each flag comes with one image made of two panels of the same 40 um field: LEFT = DAPI (nuclear stain) in grayscale,
RIGHT = composite with the membrane/boundary stain in red and DAPI in blue. Overlays on both panels: yellow outlines =
the current cell masks; green box = the flagged structure; cyan outline = the flagged cell itself (doublet / empty flags).
Nuclei here are roughly 5-9 um across; a cell mask is usually one nucleus plus a thin rim of cytoplasm.

Important calibration: the reference segmentation for this slide counts DIM nuclei as cells. In this FFPE section about
half of the real nuclei are dim, diffuse or slightly out of focus rather than bright and crisp, and the masks that are
missing are mostly exactly those dim nuclei. Judge by size, shape and territory, not by brightness.

Flag kinds and the only actions allowed for each:
- missed (green box = DAPI signal that no mask covers): "add" if the box holds a nucleus-sized DAPI blob (roughly 3-9 um
  across, at least ~8 um2) that occupies its own space, i.e. a gap between the yellow outlines, even if it is dim; also
  add when such a blob sits in its own membrane-bounded red territory with no yellow outline. "skip" only if the signal is
  a sliver of a nucleus that is already inside a yellow outline, an image-edge fragment (black padding next to it), pure
  background with no DAPI at all, or clearly smaller than a nucleus.
- doublet (cyan cell, green box = extra DAPI away from the cell's own nucleus): the reference splits cells with an
  interior RNA stain you cannot see, so a true doublet here usually looks like ONE large, elongated or bilobed DAPI
  region, not two crisp nuclei. "split" if the cyan outline holds two DAPI centres (two brightness maxima, or a waist /
  constriction between lobes), or a red membrane line crosses the cell, or the cell is clearly larger than its
  neighbours with DAPI filling both halves. "skip" if there is one compact nucleus with internal texture only, or the
  green box is background.
- empty (cyan cell, little DAPI inside): "drop" the cell if its outline contains no nucleus at all (background, a bit of
  cytoplasm, or just a membrane fragment); "skip" if there is a dim or partial nucleus inside, or the cell plainly
  belongs to a cell whose nucleus sits just outside the outline (in dense epithelium, a cell section may miss its nucleus).

Give the action you actually believe; do not default to "skip". Confidence is your probability that the action is correct.
Return JSON only: {"decisions": [{"flag_id": "<id exactly as given>", "action": "...", "confidence": 0-1, "reason": "<= 12 words"}]},
one entry per flag, in the order given."""

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "flag_id": {"type": "string"},
                    "action": {"type": "string", "enum": ["add", "split", "drop", "skip"]},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["flag_id", "action", "confidence", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------- rendering

def _window(cy: float, cx: float, size: int, H: int, W: int) -> tuple[int, int]:
    """Top-left corner of a size x size window centred on (cy, cx); may run off the image (zero-padded later)."""
    return int(round(cy - size / 2)), int(round(cx - size / 2))


def _crop_pad(a: np.ndarray, y0: int, x0: int, size: int) -> np.ndarray:
    """a[y0:y0+size, x0:x0+size] with zero padding outside the image, so the scale is constant at ROI edges."""
    H, W = a.shape[:2]
    out = np.zeros((size, size) + a.shape[2:], dtype=a.dtype)
    ys, ye = max(y0, 0), min(y0 + size, H); xs, xe = max(x0, 0), min(x0 + size, W)
    if ye > ys and xe > xs:
        out[ys - y0 : ye - y0, xs - x0 : xe - x0] = a[ys:ye, xs:xe]
    return out


def flag_centre(kind: str, row) -> tuple[float, float]:
    if kind == "missed":
        return float(row.cy), float(row.cx)
    if kind == "doublet":
        return (row.cell_y0 + row.cell_y1) / 2, (row.cell_x0 + row.cell_x1) / 2
    return (row.y0 + row.y1) / 2, (row.x0 + row.x1) / 2  # empty: the cell's own box


def render_flag(kind: str, row, rgb: np.ndarray, cells: np.ndarray, pixel_size_um: float,
                size_um: float = 40.0, scale: int = 2, label: str = "") -> np.ndarray:
    """Two-panel uint8 image (size*scale, 2*size*scale, 3): DAPI gray | R-boundary/B-DAPI composite, both with the
    current cell outlines (yellow), the flag box (green) and, for doublet/empty, the flagged cell's outline (cyan)."""
    size = int(round(size_um / pixel_size_um))
    cy, cx = flag_centre(kind, row)
    y0, x0 = _window(cy, cx, size, *cells.shape)
    comp = _crop_pad(rgb, y0, x0, size)
    lab = _crop_pad(cells.astype(np.int64), y0, x0, size)
    gray = np.repeat(comp[..., 2:3], 3, axis=-1)
    S = size * scale
    panels = []
    for base in (gray, comp):
        im = Image.fromarray(base).resize((S, S), Image.BILINEAR)
        a = np.array(im)
        lab_up = np.array(Image.fromarray(lab.astype(np.int32)).resize((S, S), Image.NEAREST))
        a[find_boundaries(lab_up, mode="inner")] = YELLOW
        if kind in ("doublet", "empty"):
            a[find_boundaries(np.where(lab_up == int(row.cell), lab_up, 0), mode="inner")] = CYAN
        im = Image.fromarray(a); d = ImageDraw.Draw(im)
        if kind in ("missed", "doublet"):  # green box = the flagged DAPI structure
            by0, by1, bx0, bx1 = (int(row.y0) - y0) * scale, (int(row.y1) - y0) * scale, (int(row.x0) - x0) * scale, (int(row.x1) - x0) * scale
            d.rectangle([bx0 - 2, by0 - 2, bx1 + 2, by1 + 2], outline=GREEN, width=2)
        panels.append(im)
    out = Image.new("RGB", (2 * S + 4, S), (40, 40, 40))
    out.paste(panels[0], (0, 0)); out.paste(panels[1], (S + 4, 0))
    if label:
        ImageDraw.Draw(out).text((4, 2), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return np.array(out)


def encode_png(img: np.ndarray) -> str:
    """Deterministic PNG -> base64 (fixed compression, no metadata) so identical crops give identical cache keys."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", compress_level=6)
    return base64.b64encode(buf.getvalue()).decode()


def flag_text(kind: str, row, pixel_size_um: float) -> str:
    um2 = pixel_size_um ** 2
    if kind == "missed":
        return f"missed: uncovered DAPI blob {row.area_px * um2:.1f} um2, {100 * row.cover:.0f}% of the whole blob under existing masks"
    if kind == "doublet":
        cell_area = (row.cell_y1 - row.cell_y0) * (row.cell_x1 - row.cell_x0) * um2
        return f"doublet: cell box ~{cell_area:.0f} um2 holds an extra DAPI blob of {row.area_px * um2:.1f} um2 away from its own nucleus"
    return f"empty: cell {row.area_px * um2:.1f} um2 with {100 * row.dapi_frac:.0f}% of its area over DAPI foreground"


# ----------------------------------------------------------------------------- planner

class MusePlanner:
    """Muse Spark triage over detector flags. See module docstring."""

    def __init__(self, batch: int = 12, kinds: tuple[str, ...] = ("missed", "doublet", "empty"), min_conf: float = 0.6,
                 model: str | None = None, reasoning_effort: str = "low", use_schema: bool = True, with_images: bool = True,
                 max_flags: int | None = None, save_dir: str | Path | None = None, size_um: float = 40.0, scale: int = 2,
                 max_tokens: int = 3000):
        self.batch, self.kinds, self.min_conf = batch, tuple(kinds), min_conf
        self.model, self.reasoning_effort, self.use_schema, self.with_images = model, reasoning_effort, use_schema, with_images
        self.max_flags, self.save_dir, self.size_um, self.scale, self.max_tokens = max_flags, save_dir, size_um, scale, max_tokens
        self.log: list[dict] = []
        self.cost_usd = 0.0; self.n_calls = 0; self.n_cached = 0; self.failures: list[dict] = []; self.budget_exceeded = False

    # -- one batch -------------------------------------------------------------------------------------------------
    def _ask(self, items: list[dict], batch_no: int) -> tuple[dict[str, dict], dict]:
        """items = [dict(flag_id, kind, text, png)] -> ({flag_id: decision}, call info)."""
        content: list[dict] = [dict(type="text", text=f"{len(items)} flags follow. Decide each one.")]
        for i, it in enumerate(items, start=1):
            content.append(dict(type="text", text=f"Flag {i}, flag_id \"{it['flag_id']}\": {it['text']}"))
            if self.with_images and it.get("png"):
                content.append(dict(type="image_url", image_url=dict(url="data:image/png;base64," + it["png"])))
        msgs = [dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=content)]
        fmt = {"type": "json_schema", "json_schema": {"name": "qc_decisions", "schema": DECISION_SCHEMA}} if self.use_schema else None
        out = None
        for max_tokens in (self.max_tokens, 2 * self.max_tokens):
            out = muse.chat(msgs, model=self.model, reasoning_effort=self.reasoning_effort, max_tokens=max_tokens,
                            response_format=fmt, tag=f"qc_plan_b{batch_no}")
            if out["finish_reason"] != "length":
                break
        self.n_calls += 1; self.n_cached += int(out["cached"]); self.cost_usd += out["cost_usd"]
        raw = out["content"] or ""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(m.group(0)) if m else {}
        decisions: dict[str, dict] = {}
        for d in parsed.get("decisions", []) if isinstance(parsed, dict) else []:
            fid = str(d.get("flag_id", ""))
            if fid in decisions:
                continue
            decisions[fid] = d
        info = dict(batch=batch_no, n=len(items), cost_usd=out["cost_usd"], cached=out["cached"], finish_reason=out["finish_reason"],
                    usage=out.get("usage"), sec=out.get("sec"), raw=raw if not decisions else None)
        return decisions, info

    # -- plan ------------------------------------------------------------------------------------------------------
    def plan(self, flags: dict[str, pd.DataFrame], context: dict) -> list[dict]:
        rgb, cells, px, seed_box = context["rgb"], context["cells"], context["pixel_size_um"], context["seed_box"]
        rnd = context.get("round", 0)
        if rnd > 0:  # one Muse pass per run: re-judging after repairs would re-bill the same crops with new masks
            return []
        items = []
        for kind in self.kinds:
            df = flags.get(kind)
            if df is None or not len(df):
                continue
            if self.max_flags is not None:  # cap per kind, so a capped run still sees every kind
                df = df.iloc[: self.max_flags]
            for idx, row in zip(df.index, df.itertuples(index=False)):
                fid = f"{kind}:{idx}"
                png = encode_png(render_flag(kind, row, rgb, cells, px, self.size_um, self.scale, label=fid)) if self.with_images else None
                items.append(dict(flag_id=fid, kind=kind, row=row, text=flag_text(kind, row, px), png=png))
        if self.save_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        acts: list[dict] = []
        for b in range(0, len(items), self.batch):
            chunk = items[b : b + self.batch]; batch_no = b // self.batch
            if self.save_dir and self.with_images:
                tiles = [np.array(Image.open(io.BytesIO(base64.b64decode(it["png"])))) for it in chunk]
                Image.fromarray(np.concatenate(tiles, axis=0)).save(Path(self.save_dir) / f"batch_{batch_no:02d}.png")
            try:
                decisions, info = self._ask(chunk, batch_no)
            except muse.BudgetExceeded as e:
                warnings.warn(f"MusePlanner: {e}; returning the {len(acts)} actions decided so far")
                self.budget_exceeded = True; self.failures.append(dict(batch=batch_no, error=str(e)))
                break
            except Exception as e:  # noqa: BLE001  (API / parse errors: skip this batch, keep going)
                warnings.warn(f"MusePlanner: batch {batch_no} failed ({e}); skipping it")
                self.failures.append(dict(batch=batch_no, error=repr(e)))
                continue
            if not decisions:
                self.failures.append(dict(batch=batch_no, error="no decisions parsed", raw=info.get("raw")))
            per_flag_cost = info["cost_usd"] / max(len(chunk), 1)
            for it in chunk:
                d = decisions.get(it["flag_id"])
                action, conf, reason = "skip", 0.0, "no decision returned"
                if d is not None:
                    action = str(d.get("action", "skip")); reason = str(d.get("reason", ""))[:200]
                    try:
                        conf = float(d.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        conf = 0.0
                    if action not in ALLOWED[it["kind"]]:
                        reason = f"invalid reply: {action} for {it['kind']}"; action = "skip"
                accepted = action != "skip" and conf >= self.min_conf
                row = it["row"]
                entry = dict(kind=it["kind"], flag_id=it["flag_id"], cell=int(getattr(row, "cell", 0) or 0),
                             cy=float(flag_centre(it["kind"], row)[0]), cx=float(flag_centre(it["kind"], row)[1]),
                             decision=action, confidence=conf, reason=reason, accepted=bool(accepted),
                             batch=batch_no, cached=bool(info["cached"]), cost_usd=per_flag_cost)
                self.log.append(entry)
                if not accepted:
                    continue
                if action == "add":
                    acts.append(dict(action="add", box=(int(row.y0), int(row.y1), int(row.x0), int(row.x1)), flag_id=it["flag_id"]))
                elif action == "split":
                    sb = seed_box.get(int(row.cell))
                    if sb is None:
                        entry.update(accepted=False, reason=reason + " (no seed box)"); continue
                    acts.append(dict(action="split", cell=int(row.cell), boxes=[sb, (int(row.y0), int(row.y1), int(row.x0), int(row.x1))], flag_id=it["flag_id"]))
                elif action == "drop":
                    acts.append(dict(action="drop", cell=int(row.cell), flag_id=it["flag_id"]))
        # adds largest-first, like RulesPlanner, so a big nucleus is not blocked by a sliver added before it
        order = {"drop": 0, "split": 1, "add": 2}
        acts.sort(key=lambda a: (order[a["action"]], -((a["box"][1] - a["box"][0]) * (a["box"][3] - a["box"][2])) if a["action"] == "add" else 0))
        return acts

    def summary(self) -> dict:
        df = pd.DataFrame(self.log)
        per_kind = {}
        if len(df):
            for k, g in df.groupby("kind"):
                per_kind[k] = dict(n=int(len(g)), accepted=int(g.accepted.sum()), mean_conf=float(g.confidence.mean()))
        return dict(n_flags=int(len(df)), n_calls=self.n_calls, n_cached=self.n_cached, cost_usd=self.cost_usd,
                    failures=self.failures, budget_exceeded=self.budget_exceeded, per_kind=per_kind)
