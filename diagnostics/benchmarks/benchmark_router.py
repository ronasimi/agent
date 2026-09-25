#!/usr/bin/env python3
"""Measure real router prefill/cache latency with the configured tool catalog."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ollama import Client

from al_agent.model_residency import background_inference_slot
from al_agent.system1_router import SystemOneRouter
from tools.catalog import catalog_snapshot
from tools.config import load_config

REQUESTS = (
    "Check the current local time in America/Toronto.",
    "What is the weather forecast for London Ontario?",
    "Calculate 19 times 23.",
    "List the connected devices on my local network.",
    "Read the current host CPU temperature and memory use.",
)


def summarize(rows: list[dict]) -> dict:
    wall = sorted(r["decision_wall_ms"] for r in rows if not r["router_error"])
    metrics = ("load_duration", "prompt_eval_duration", "eval_duration")
    summary = {
        "runs": len(rows),
        "errors": sum(bool(r["router_error"]) for r in rows),
        "selected": sum(bool(r["selected"]) for r in rows),
        "fallbacks": sum(r["tier"] == "fallback" for r in rows),
        "median_decision_ms": statistics.median(wall) if wall else None,
        "p95_decision_ms": wall[max(0, math.ceil(len(wall) * 0.95) - 1)]
        if wall
        else None,
    }
    for name in metrics:
        values = [
            r["metrics"][name] / 1e6 for r in rows if r["metrics"].get(name) is not None
        ]
        summary["median_" + name.replace("_duration", "_ms")] = (
            statistics.median(values) if values else None
        )
    cached = [
        r["metrics"]["prompt_eval_cached_count"]
        for r in rows
        if r["metrics"].get("prompt_eval_cached_count") is not None
    ]
    summary["median_cached_tokens"] = statistics.median(cached) if cached else None
    return summary


def main() -> int:
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--cold",
        action="store_true",
        help="Unload only the router before measuring its initial warmup.",
    )
    args = parser.parse_args()
    cfg = load_config()["agent"]
    rcfg = cfg["router"]
    client = Client(host=cfg["host"], timeout=rcfg["timeout_seconds"])
    warm_client = Client(host=cfg["host"], timeout=rcfg["warmup_timeout_seconds"])
    events = []
    kwargs = dict(
        model=rcfg["model"],
        options=rcfg["options"],
        keep_alive=rcfg["keep_alive"],
        candidate_count=rcfg["candidates"],
        route_threshold=rcfg["route_threshold"],
        prefix_max_bytes=rcfg["prefix_max_bytes"],
        description_chars=rcfg["description_chars"],
        inference_slot=background_inference_slot,
        on_metrics=events.append,
    )
    router = SystemOneRouter(client, **kwargs)
    warm_router = SystemOneRouter(warm_client, **{**kwargs, "inference_slot": nullcontext})
    schemas, _, metadata = catalog_snapshot()
    # Keep explicit unloading and initial prefill atomic relative to the Web UI
    # maintenance thread, so it cannot warm the model between these two calls.
    with background_inference_slot():
        if args.cold:
            client.generate(model=router.model, keep_alive=0)
        warmed = warm_router.warm(schemas)
    warm_event = events[-1] if events else None
    rows = []
    for i in range(max(1, args.runs)):
        query = REQUESTS[i % len(REQUESTS)]
        started = time.monotonic()
        decision = router.decide(query, schemas, metadata)
        rows.append(
            {
                "query": query,
                "selected": decision.selected,
                "tier": decision.tier,
                "confidence": decision.confidence,
                "raw_choice": decision.raw_choice,
                "router_error": decision.router_error,
                "metrics": decision.metrics,
                "decision_wall_ms": round((time.monotonic() - started) * 1000, 3),
            }
        )
        router.reset_turn()
    print(
        json.dumps(
            {
                "model": router.model,
                "options": router.options,
                "cold_requested": args.cold,
                "warmup_ok": warmed,
                "warmup": {
                    k: v for k, v in (warm_event or {}).items() if k != "prompt"
                },
                "summary": summarize(rows),
                "runs": rows,
            },
            indent=2,
        )
    )
    return 1 if not warmed or any(r["router_error"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
