#!/usr/bin/env python3
"""Measure Ollama cold/preload/prefix-prime behavior with real harness settings.

This benchmark is intentionally opt-in and does not mutate harness state. It
prints one JSON record per scenario so cache/load behavior can be compared on
the actual host/model instead of inferred from prompt-template assumptions.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from ollama import Client

from tools.config import load_config
from tools.catalog import get_tool_schema, load_tools
from al_agent.prompts import build_system_prompt


def _field(obj: Any, name: str, default: Any = 0) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _unload(client: Client, model: str, options: dict[str, Any]) -> None:
    try:
        client.chat(model=model, messages=[], options=options, keep_alive=0)
    except Exception:
        pass


def _probe(client: Client, model: str, options: dict[str, Any], system_prompt: str, tools: list[dict]) -> dict[str, Any]:
    started = time.monotonic()
    first = None
    final: Any = {}
    stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Reply with exactly: warmup benchmark"},
        ],
        tools=tools,
        options={**options, "num_predict": 12},
        keep_alive=-1,
        think=False,
        stream=True,
    )
    for chunk in stream:
        if first is None:
            first = time.monotonic()
        final = chunk
    elapsed = time.monotonic() - started
    evaluated = int(_field(final, "prompt_eval_count", 0) or 0)
    cached = int(_field(final, "prompt_eval_cached_count", 0) or 0)
    denom = evaluated + cached
    return {
        "wall_ms": round(elapsed * 1000, 2),
        "ttft_ms": round(((first or time.monotonic()) - started) * 1000, 2),
        "load_ms": round(float(_field(final, "load_duration", 0) or 0) / 1_000_000, 2),
        "prompt_eval_ms": round(float(_field(final, "prompt_eval_duration", 0) or 0) / 1_000_000, 2),
        "prompt_eval_count": evaluated,
        "prompt_eval_cached_count": cached,
        "cache_hit_pct": round((cached * 100.0 / denom) if denom else 0.0, 2),
        "eval_count": int(_field(final, "eval_count", 0) or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--settle", type=float, default=0.35, help="seconds after unload/load requests")
    args = parser.parse_args()

    cfg = load_config().get("agent", {})
    model = args.model or str(cfg.get("model") or "agent-main:4b")
    host = args.host or str(cfg.get("host") or "http://127.0.0.1:11434")
    options = dict(cfg.get("main_options") or {})
    system_prompt = build_system_prompt()
    load_tools()
    tool_names = ["current_time", "hostname"]
    tools = [schema for schema in (get_tool_schema(name) for name in tool_names) if schema]
    client = Client(host=host)

    scenarios = ("cold", "weights_preloaded", "system_prefix_primed")
    for scenario in scenarios:
        _unload(client, model, options)
        time.sleep(max(0.0, args.settle))
        if scenario in {"weights_preloaded", "system_prefix_primed"}:
            client.chat(model=model, messages=[], options=options, keep_alive=-1)
        if scenario == "system_prefix_primed":
            client.chat(
                model=model,
                messages=[{"role": "system", "content": system_prompt}],
                options={**options, "num_predict": 1},
                keep_alive=-1,
                think=False,
                stream=False,
            )
        time.sleep(max(0.0, args.settle))
        result = _probe(client, model, options, system_prompt, tools)
        print(json.dumps({"scenario": scenario, "model": model, **result}, sort_keys=True))

    print(json.dumps({
        "interpretation": (
            "Prefer weight preloading. Enable warmup.prime_system_prefix only when the primed scenario "
            "shows reproducible prompt_eval_cached_count/cache_hit_pct improvement for real tool-bearing turns."
        )
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
