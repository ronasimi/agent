#!/usr/bin/env python3
"""Benchmark 0.8B vs 2B on the same terminal-classification task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from ollama import Client
from al_agent.micro_model import validate_completion_with_model

CASES = [
    ("Check CPU, RAM, and disk pressure on this host.", "host_snapshot ok; pressure_snapshot ok; filesystem_snapshot ok"),
    ("Why can't api.example.com resolve?", "dns_diagnose failed: SERVFAIL; endpoint_probe not run"),
    ("Read the config and summarize it.", "read_text ok; requested file contents present"),
    ("Get the current WTI price.", "market_quote ok; WTI quote and provider timestamp present"),
    ("Run the tests and fix failures.", "repo_checks failed; traceback indicates one test failure"),
    ("What time is it?", "current_time ok; exact local time present"),
]


def _pct95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _run(client, model, cases, runs, options, keep_alive):
    all_values: list[float] = []
    per_case = {}
    failed = 0
    for request, state in cases:
        values = []
        outcomes = {"terminal": 0, "fallback": 0}
        for _ in range(runs):
            started = time.monotonic()
            result = validate_completion_with_model(
                client,
                model,
                request,
                state,
                signal={"kind": "benchmark"},
                stage="final",
                options=options,
                keep_alive=keep_alive,
            )
            elapsed = (time.monotonic() - started) * 1000.0
            values.append(elapsed)
            all_values.append(elapsed)
            if result is None:
                outcomes["fallback"] += 1
            else:
                outcomes["terminal"] += 1
        per_case[request] = {
            "runs": len(values),
            "median_ms": round(statistics.median(values), 2),
            "p95_ms": round(_pct95(values), 2),
            **outcomes,
        }
    return {
        "model": model,
        "runs": len(all_values),
        "failed_transport_runs": failed,
        "median_ms": round(statistics.median(all_values), 2) if all_values else None,
        "p95_ms": round(_pct95(all_values), 2) if all_values else None,
        "cases": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--compare-fast", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    agent = dict(cfg.get("agent") or {})
    client = Client(host=str(agent.get("host") or "http://127.0.0.1:11434"), timeout=60)

    micro_model = str(agent.get("micro_model") or "agent-micro:0.8b")
    micro_options = dict(agent.get("micro_options") or {})
    micro_keep_alive = agent.get("micro_model_keep_alive", "2m")

    warm_started = time.monotonic()
    client.chat(model=micro_model, messages=[], options=micro_options, keep_alive=micro_keep_alive, stream=False)
    warm_ms = (time.monotonic() - warm_started) * 1000.0

    output = {
        "micro_model": _run(client, micro_model, CASES, max(1, args.runs), micro_options, micro_keep_alive),
        "warm_ms": round(warm_ms, 2),
    }

    if args.compare_fast:
        fast_model = str(agent.get("fast_model") or "agent-fast:2b")
        fast_options = dict(agent.get("fast_options") or {})
        fast_keep_alive = agent.get("fast_model_keep_alive", "2m")
        output["fast_model"] = _run(client, fast_model, CASES, max(1, args.runs), fast_options, fast_keep_alive)
        mm = output["micro_model"]["median_ms"]
        fm = output["fast_model"]["median_ms"]
        output["median_ms_saved_by_micro"] = round(float(fm) - float(mm), 2) if mm is not None and fm is not None else None

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
