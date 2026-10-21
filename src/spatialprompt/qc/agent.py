"""QC agent loop: detect -> plan -> repair -> re-detect.

The planner decides which flags become which actions. `RulesPlanner` is deterministic and
runs today; `MusePlanner` (Muse Spark via the Meta Model API, tool-calling) drops in behind the
same interface once MODEL_API_KEY exists. Both receive the flag tables as data and return a
list of action dicts; the loop executes them with qc.repair and records every step.

    agent = QCAgent(seg, planner=RulesPlanner())
    cells2, log = agent.run(rgb, cells, seeds, dapi, pixel_size_um, rounds=1)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from spatialprompt.qc import detectors, repair


class RulesPlanner:
    """Fixed policy: add every missed nucleus (largest first), split doublets, merge splits, drop empties."""

    def __init__(self, do_add: bool = True, do_split: bool = True, do_merge: bool = False, do_drop: bool = False,
                 max_actions: int = 2000):
        self.do = dict(add=do_add, split=do_split, merge=do_merge, drop=do_drop)
        self.max_actions = max_actions

    def plan(self, flags: dict[str, pd.DataFrame], context: dict) -> list[dict]:
        acts: list[dict] = []
        seed_box = context["seed_box"]  # cell id -> (y0, y1, x0, x1) of its seed
        if self.do["drop"]:
            acts += [dict(action="drop", cell=int(r.cell)) for r in flags["empty"].itertuples()]
        if self.do["merge"]:
            acts += [dict(action="merge", cells=list(map(int, r.cells))) for r in flags["split"].itertuples()]
        if self.do["split"]:
            for r in flags["doublet"].itertuples():
                sb = seed_box.get(int(r.cell))
                if sb is None:
                    continue
                acts.append(dict(action="split", cell=int(r.cell), boxes=[sb, (int(r.y0), int(r.y1), int(r.x0), int(r.x1))]))
        if self.do["add"]:
            m = flags["missed"].sort_values("area_px", ascending=False)
            acts += [dict(action="add", box=(int(r.y0), int(r.y1), int(r.x0), int(r.x1))) for r in m.itertuples()]
        return acts[: self.max_actions]


class MusePlanner:
    """Placeholder for the Muse Spark planner: same interface, asks the model to choose actions
    from the flag tables (as JSON) with a tool schema. Implemented when the API key is available."""

    def plan(self, flags, context):  # pragma: no cover
        raise NotImplementedError("MusePlanner needs MODEL_API_KEY; use RulesPlanner for now")


def seed_boxes(seeds: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    out = {}
    for i, sl in enumerate(ndi.find_objects(np.asarray(seeds).astype(np.int64)), start=1):
        if sl is not None:
            out[i] = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
    return out


@dataclass
class QCAgent:
    seg: object                      # spatialprompt.segment.Segmenter (for prompt_boxes)
    planner: object = field(default_factory=RulesPlanner)
    dilate_um: float = 2.0
    crop: int = 384
    # defaults from the roi01/roi03 sweep (2026-10-21): 0.45 x Otsu and a 120 um2 cap gave the best F1;
    # gating on SAM's predicted IoU (0.8 / 0.9) removed nearly every add and gained nothing
    low_otsu: float = 0.45           # permissive DAPI threshold for the missed detector
    min_pred_iou: float = 0.0        # accept an added cell only if SAM's predicted IoU is at least this
    max_area_um2: float | None = 120.0  # reject an added cell larger than this

    def run(self, rgb: np.ndarray, cells: np.ndarray, seeds: np.ndarray, dapi: np.ndarray, pixel_size_um: float,
            rounds: int = 1, fg: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        d = int(round(self.dilate_um / pixel_size_um))
        cells = np.asarray(cells).astype(np.int64).copy()
        seeds = np.asarray(seeds).astype(np.int64)
        if fg is None:
            fg = detectors.dapi_foreground(dapi, pixel_size_um)
        fg_low = detectors.dapi_foreground(dapi, pixel_size_um, otsu_scale=self.low_otsu)
        sb = seed_boxes(seeds)
        log: dict = dict(rounds=[], n_before=int(len(np.unique(cells)) - 1))
        for r in range(rounds):
            flags = detectors.detect_all(cells, seeds, fg, pixel_size_um, fg_low=fg_low)
            summ = detectors.summarize(flags, len(np.unique(cells)) - 1)
            acts = self.planner.plan(flags, dict(pixel_size_um=pixel_size_um, round=r, seed_box=sb))
            recs = []
            for a in acts:
                if a["action"] == "drop":
                    cells, rec = repair.drop_cell(cells, a["cell"])
                elif a["action"] == "merge":
                    cells, rec = repair.merge_cells(cells, a["cells"])
                elif a["action"] == "split":
                    cells, rec = repair.split_cell(self.seg, rgb, cells, a["cell"], a["boxes"], d, self.crop)
                    if rec.get("ok"):  # the new half gets its own seed box = the extra-nucleus box
                        for nid, box in zip(rec["new_ids"][1:], a["boxes"][1:]):
                            sb[nid] = box
                elif a["action"] == "add":
                    cells, rec = repair.add_cell(self.seg, rgb, cells, a["box"], d, self.crop, min_pred_iou=self.min_pred_iou,
                                                 max_area=None if self.max_area_um2 is None else int(self.max_area_um2 / pixel_size_um ** 2))
                    if rec.get("ok"):
                        sb[rec["new_id"]] = a["box"]
                        seeds = seeds.copy(); y0, y1, x0, x1 = a["box"]
                        win = seeds[y0:y1, x0:x1]; win[(win == 0) & fg[y0:y1, x0:x1]] = rec["new_id"]
                else:
                    rec = dict(action=a["action"], ok=False, reason="unknown action")
                recs.append(rec)
            done = pd.DataFrame(recs) if recs else pd.DataFrame(columns=["action", "ok"])
            per = done.groupby("action").ok.agg(["count", "sum"]).rename(columns={"count": "planned", "sum": "applied"}).astype(int).to_dict("index") if len(done) else {}
            log["rounds"].append(dict(round=r, flags=summ, actions=per, n_after=int(len(np.unique(cells)) - 1)))
            if not len(done) or not done.ok.any():
                break
        log["n_after"] = int(len(np.unique(cells)) - 1); log["sec"] = time.time() - t0
        return cells.astype(np.uint32), log


def dump_log(log: dict) -> str:
    return json.dumps(log, indent=2, default=float)
