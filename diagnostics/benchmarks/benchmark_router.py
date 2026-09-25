#!/usr/bin/env python3
"""Measure deterministic tool-candidate retrieval latency and selections."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from al_agent.deterministic_router import DeterministicToolRouter
from tools.catalog import catalog_snapshot
from tools.config import load_config

REQUESTS = (
    "Check the current local time in America/Toronto.",
    "What is the weather forecast for London Ontario?",
    "Calculate 19 times 23.",
    "List the connected devices on my local network.",
    "Read the current host CPU temperature and memory use.",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config()["agent"]
    rcfg = cfg["tool_routing"]
    router = DeterministicToolRouter(
        candidate_count=rcfg["candidate_limit"],
        auto_activate_threshold=rcfg["auto_activate_threshold"],
        auto_activate_margin=rcfg["auto_activate_margin"],
        min_candidate_score=rcfg["min_candidate_score"],
    )
    schemas, _, metadata = catalog_snapshot()
    rows = []
    for i in range(max(1, args.runs)):
        query = REQUESTS[i % len(REQUESTS)]
        started = time.perf_counter()
        decision = router.decide(query, schemas, metadata)
        elapsed = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "query": query,
                "selected": decision.selected,
                "tier": decision.tier,
                "confidence": decision.confidence,
                "decision_wall_ms": elapsed,
            }
        )
    wall = sorted(row["decision_wall_ms"] for row in rows)
    p95 = wall[max(0, math.ceil(len(wall) * 0.95) - 1)]
    print(
        json.dumps(
            {
                "mode": "deterministic_catalog_prefilter",
                "separate_model": False,
                "llm_calls": 0,
                "config": rcfg,
                "summary": {
                    "runs": len(rows),
                    "median_decision_ms": round(statistics.median(wall), 4),
                    "p95_decision_ms": round(p95, 4),
                },
                "sample": rows[: len(REQUESTS)],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
