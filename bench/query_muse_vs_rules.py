"""Muse Spark query parser vs the rules backend on the demo queries.

Every Muse call goes through spatialprompt.muse (disk cache + per-key spend cap), so re-running is free
and a run can never take the key past MUSE_BUDGET_USD. Worst case at standard rates is printed first.

    python -X utf8 bench/query_muse_vs_rules.py              # all queries
    python -X utf8 bench/query_muse_vs_rules.py --n 2        # first two only
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spatialprompt import muse
from spatialprompt.query import parse_query

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    "CD8+ T cells at the tumor boundary",
    "CD8+ T cells inside the tumor",
    "tumor cells",
    "macrophages",
    "B cells within 30 um of vessels",
    "HER2+ tumor cells",
    "regulatory T cells near the tumor",
    "CD4+ T cells that are FOXP3+",
    "proliferating tumor cells",
    "exhausted CD8 T cells in the stroma",
    "endothelial cells",
    "fibroblasts away from the tumor",
]


def groups(g) -> str:
    return " & ".join("(" + "|".join(sorted(x)) + ")" for x in sorted(map(sorted, g or [])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="first n queries only (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "results" / "query_muse_vs_rules.csv"))
    args = ap.parse_args()
    qs = QUERIES[: args.n] if args.n else QUERIES
    panel = list(np.load(ROOT / "data" / "roi" / "roi01" / "counts_10x_genes.npy", allow_pickle=True).astype(str))

    muse.load_env()
    model = os.environ.get("MUSE_MODEL", muse.DEFAULT_MODEL)
    worst = len(qs) * (2500 * muse.prices(model)[0] + 1500 * muse.prices(model)[2]) / 1e6
    print(f"{len(qs)} queries on {model}; worst case ${worst:.3f} if nothing is cached")
    print(muse.summary())

    rows = []
    for t in qs:
        r = parse_query(t, panel, backend="rules")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = parse_query(t, panel, backend="muse")
        row = dict(query=t,
                   rules_pos=groups(r["positive"]), muse_pos=groups(m["positive"]),
                   rules_neg=groups(r["negative"]), muse_neg=groups(m["negative"]),
                   rules_min=r["min_count"], muse_min=m["min_count"],
                   rules_region=(r["region"] or {}).get("name"), muse_region=(m["region"] or {}).get("name"),
                   rules_spatial=json.dumps(r["spatial"]), muse_spatial=json.dumps(m["spatial"]),
                   muse_sam_prompt=m["sam_prompt"], muse_dropped="; ".join(str(x.message) for x in w),
                   cost_usd=m["_cost_usd"], cached=m["_cached"], muse_json=json.dumps({k: v for k, v in m.items() if not k.startswith("_")}))
        price = "cached" if m["_cached"] else f"${m['_cost_usd']:.4f}"
        row["same_markers"] = row["rules_pos"] == row["muse_pos"] and row["rules_neg"] == row["muse_neg"]
        row["same_spatial"] = row["rules_region"] == row["muse_region"] and \
            (r["spatial"] or {}).get("relation") == (m["spatial"] or {}).get("relation")
        rows.append(row)
        print(f"\n{t}\n  rules: {row['rules_pos']}  not {row['rules_neg'] or '-'}  min {row['rules_min']}  "
              f"{row['rules_region']} {r['spatial']}\n  muse : {row['muse_pos']}  not {row['muse_neg'] or '-'}  "
              f"min {row['muse_min']}  {row['muse_region']} {m['spatial']}  "
              f"[{price}]"
              + (f"\n  dropped: {row['muse_dropped']}" if row["muse_dropped"] else ""))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nsame markers {df.same_markers.sum()}/{len(df)}, same region+relation {df.same_spatial.sum()}/{len(df)}; "
          f"this run ${df.cost_usd.sum():.4f}\n{muse.summary()}\nwrote {args.out}")


if __name__ == "__main__":
    main()
