"""Muse Spark (Meta Model API) client with a disk cache and a hard spend cap.

Every Muse call in the package goes through `chat()`:
  * cache: an identical request (model + messages + params) is answered from data/cache/muse/ for free.
  * budget: spend is logged per API key (a sha256 fingerprint, never the key) in
    data/logs/muse_usage.jsonl. A call whose worst case (estimated input + max_tokens of output, at the
    model's price) would take that key past MUSE_BUDGET_USD (default $1) raises BudgetExceeded before
    anything is sent. A new key starts a fresh ledger automatically.
  * reasoning cannot be switched off on Muse Spark and is billed as output, so reasoning_effort defaults
    to "low" and max_tokens (which caps reasoning + visible output together) to 1500.

Env, read from .env at the repo root: MODEL_API_KEY (or META_API_KEY), MUSE_MODEL (default
muse-spark-1.3; the -contributor ids cost ~1/12 on input and ~1/20 on output, but Meta may train on
those requests), MUSE_BUDGET_USD.

    python -m spatialprompt.muse            # spend so far per key
    python -m spatialprompt.muse --ping     # one minimal call: checks key, model id and usage fields
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "muse"
LEDGER = ROOT / "data" / "logs" / "muse_usage.jsonl"
BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
DEFAULT_BUDGET_USD = 1.0

# USD per 1M tokens: (input, cached input, output); reasoning is billed as output
# (dev.meta.ai/docs/pricing-rate-limits, 2026-10-26)
STANDARD = (1.25, 0.15, 4.25)
CONTRIBUTOR = (0.10, 0.002, 0.20)


class BudgetExceeded(RuntimeError):
    pass


def load_env() -> None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_key() -> str | None:
    load_env()
    return os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def prices(model: str) -> tuple[float, float, float]:
    if model.endswith("-contributor"):
        return CONTRIBUTOR
    if not model.startswith("muse-spark-"):
        warnings.warn(f"no price for {model}; budgeting at Muse Spark standard rates")
    return STANDARD


def budget_usd() -> float:
    load_env()
    return float(os.environ.get("MUSE_BUDGET_USD", DEFAULT_BUDGET_USD))


def ledger(fp: str | None = None) -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    return [r for r in rows if fp is None or r["key"] == fp]


def spent(fp: str) -> float:
    return sum(r["cost_usd"] for r in ledger(fp))


def _log(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _usage(u) -> tuple[int, int, int, int]:
    """(prompt, cached prompt, completion, reasoning) tokens from an OpenAI-style usage object."""
    if u is None:
        return 0, 0, 0, 0
    ptd, ctd = getattr(u, "prompt_tokens_details", None), getattr(u, "completion_tokens_details", None)
    return (int(u.prompt_tokens or 0), int(getattr(ptd, "cached_tokens", 0) or 0),
            int(u.completion_tokens or 0), int(getattr(ctd, "reasoning_tokens", 0) or 0))


IMAGE_TOKENS_EST = 400  # a 376x752 QC crop measures ~250-300 tokens; the docs say ~1,400 at 1280 px


def estimate_input(messages: list[dict], tools: list | None = None) -> tuple[int, int]:
    """(worst-case input tokens, number of images). Text at ~3 chars/token, which errs high; an
    image_url part counts a flat IMAGE_TOKENS_EST instead of its base64 length (a 50 KB PNG is
    70k chars, which would otherwise be 'budgeted' as 24k tokens and trip the cap)."""
    n_tok, n_img = 50 + len(json.dumps(tools or [])) // 3, 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n_tok += len(c) // 3
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "image_url":
                    n_img += 1; n_tok += IMAGE_TOKENS_EST
                else:
                    n_tok += len(json.dumps(part)) // 3
    return n_tok, n_img


def billed_output(completion: int, reasoning: int) -> int:
    # completion_tokens already includes reasoning_tokens: ping on 2026-10-26 returned "OK" with
    # completion 23, reasoning 13, total 38 = prompt 15 + completion 23
    return completion


def chat(messages: list[dict], *, model: str | None = None, reasoning_effort: str | None = "low",
         max_tokens: int = 1500, response_format: dict | None = None, tools: list | None = None,
         tool_choice=None, use_cache: bool = True, tag: str = "") -> dict:
    """One chat completion under the cache and the budget. Returns
    dict(content, tool_calls, finish_reason, model, usage, cost_usd, cached, sec)."""
    load_env()
    model = model or os.environ.get("MUSE_MODEL", DEFAULT_MODEL)
    params = {k: v for k, v in dict(reasoning_effort=reasoning_effort, max_tokens=max_tokens, response_format=response_format,
                                     tools=tools, tool_choice=tool_choice).items() if v is not None}
    h = hashlib.sha256(json.dumps(dict(model=model, messages=messages, **params), sort_keys=True, default=str).encode()).hexdigest()
    hit = CACHE_DIR / f"{h}.json"
    if use_cache and hit.exists():
        return {**json.loads(hit.read_text()), "cost_usd": 0.0, "cached": True}

    key = api_key()
    if not key:
        raise RuntimeError("MODEL_API_KEY not set (put it in .env at the repo root)")
    fp, cap = key_fingerprint(key), budget_usd()
    p_in, p_cached, p_out = prices(model)
    est_in, n_images = estimate_input(messages, tools)
    worst = (est_in * p_in + max_tokens * p_out) / 1e6
    so_far = spent(fp)
    if so_far + worst > cap:
        raise BudgetExceeded(f"key {fp}: spent ${so_far:.4f} of ${cap:.2f}; this call could cost up to ${worst:.4f}. "
                             "Raise MUSE_BUDGET_USD in .env to continue.")

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=os.environ.get("MODEL_API_BASE_URL", BASE_URL), max_retries=1, timeout=180)
    t0 = time.time()
    r = client.chat.completions.create(model=model, messages=messages, **params)
    sec = time.time() - t0
    n_in, n_cached, n_out, n_reason = _usage(r.usage)
    cost = ((n_in - n_cached) * p_in + n_cached * p_cached + billed_output(n_out, n_reason) * p_out) / 1e6
    choice = r.choices[0]
    if choice.finish_reason == "length":
        warnings.warn(f"Muse response truncated at max_tokens={max_tokens} (reasoning {n_reason} tokens)")
    rec = dict(content=choice.message.content, tool_calls=[tc.model_dump() for tc in choice.message.tool_calls or []],
               finish_reason=choice.finish_reason, model=r.model or model,
               usage=r.usage.model_dump() if r.usage is not None else None, sec=sec)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hit.write_text(json.dumps(rec))
    _log(dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), key=fp, model=model, tag=tag, hash=h, prompt=n_in, cached_prompt=n_cached,
              completion=n_out, reasoning=n_reason, n_images=n_images, cost_usd=cost, sec=round(sec, 2)))
    return {**rec, "cost_usd": cost, "cached": False}


def summary() -> str:
    key = api_key()
    fp = key_fingerprint(key) if key else None
    lines = [f"budget per key: ${budget_usd():.2f}   model: {os.environ.get('MUSE_MODEL', DEFAULT_MODEL)}   "
             f"current key: {fp or 'not set'}"]
    by_key: dict[str, list[dict]] = {}
    for r in ledger():
        by_key.setdefault(r["key"], []).append(r)
    for k, rows in by_key.items():
        cost = sum(r["cost_usd"] for r in rows)
        lines.append(f"  key {k}{' (current)' if k == fp else ''}: {len(rows)} calls, ${cost:.4f}, "
                     f"in {sum(r['prompt'] for r in rows)} / out {sum(r['completion'] for r in rows)} / "
                     f"reasoning {sum(r['reasoning'] for r in rows)} tokens")
    if not by_key:
        lines.append("  no billed calls yet")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--ping" in sys.argv:
        out = chat([dict(role="user", content="Reply with the single word OK.")], reasoning_effort="minimal",
                   max_tokens=300, use_cache=False, tag="ping")
        print(json.dumps({k: out[k] for k in ("model", "content", "finish_reason", "usage", "cost_usd", "sec")}, indent=2))
    print(summary())
