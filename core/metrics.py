"""
core/metrics.py
===============
E2 — cost & speed measurement.

A tiny global usage tracker. embed.py and report/generate.py report their token
usage here; the eval harness (and anyone else) reads it to compute $/report and
tokens/report. Reset per-run with reset().

Pricing (USD per 1M tokens) — update if model prices change. These are the
defaults for the models Aletheon uses; override via set_pricing() if needed.
"""

import time

# USD per 1,000,000 tokens (approximate list prices; adjust as needed).
PRICING = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

_state = {
    "embed_tokens": 0,
    "llm_input_tokens": 0,
    "llm_output_tokens": 0,
    "embed_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "t_start": None,
    "t_end": None,
}


def reset():
    _state.update({
        "embed_tokens": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "t_start": time.time(),
        "t_end": None,
    })


def stop_timer():
    _state["t_end"] = time.time()


def record_embed(tokens: int, model: str = None):
    _state["embed_tokens"] += int(tokens or 0)
    if model:
        _state["embed_model"] = model


def record_llm(input_tokens: int, output_tokens: int, model: str = None):
    _state["llm_input_tokens"] += int(input_tokens or 0)
    _state["llm_output_tokens"] += int(output_tokens or 0)
    if model:
        _state["llm_model"] = model


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]


def summary() -> dict:
    """Return cost + speed for the current run."""
    embed_cost = _cost(_state["embed_model"], _state["embed_tokens"], 0)
    llm_cost = _cost(_state["llm_model"], _state["llm_input_tokens"], _state["llm_output_tokens"])
    elapsed = None
    if _state["t_start"] is not None:
        end = _state["t_end"] or time.time()
        elapsed = end - _state["t_start"]
    return {
        "embed_tokens": _state["embed_tokens"],
        "llm_input_tokens": _state["llm_input_tokens"],
        "llm_output_tokens": _state["llm_output_tokens"],
        "embed_cost_usd": round(embed_cost, 6),
        "llm_cost_usd": round(llm_cost, 6),
        "total_cost_usd": round(embed_cost + llm_cost, 6),
        "elapsed_sec": round(elapsed, 2) if elapsed is not None else None,
    }
