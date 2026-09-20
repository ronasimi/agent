#!/usr/bin/env python3
"""Benchmark the optional Laya decision layer on the deployment host.

Run inside the agent container after the Laya checkpoint has downloaded:
    python scripts/benchmark_laya.py --runs 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from al_agent.decision_engine import DecisionEngineClient, route_with_fast_model
from tools.config import load_config

PROMPTS = [
    "Explain the difference between TCP and UDP.",
    "Check CPU, RAM, and disk pressure on this host.",
    "Why can't api.example.com resolve from this machine?",
    "Find TODO comments in the Python files in this repository.",
    "What are the latest local headlines?",
    "What else can you tell me about that?",
    "Remind me tomorrow morning to review the report.",
    "Compare these two JSON files and summarize the changed keys.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--compare-fast", action="store_true", help="Also benchmark the configured 2B Ollama routing fallback")
    args = parser.parse_args()
    cfg = dict((load_config().get("agent") or {}).get("decision_engine") or {})
    cfg["enabled"] = True
    cfg.setdefault("training_capture", {})["enabled"] = False
    engine = DecisionEngineClient(cfg)
    started = time.monotonic()
    engine.preload(timeout=0.5)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not engine.health().get("loaded"):
        time.sleep(0.25)
    cold_ms = (time.monotonic() - started) * 1000
    if not engine.health().get("loaded"):
        print(json.dumps({"ok": False, "error": "Laya sidecar did not become ready", "health": engine.health()}, indent=2))
        return 2
    samples = []
    by_prompt = {}
    for prompt in PROMPTS:
        vals = []
        for _ in range(max(1, args.runs)):
            result = engine.route_turn(prompt)
            if result.ok:
                vals.append(result.latency_ms)
                samples.append(result.latency_ms)
        by_prompt[prompt] = {
            "runs": len(vals),
            "median_ms": round(statistics.median(vals), 2) if vals else None,
            "p95_ms": round(sorted(vals)[max(0, int(len(vals) * .95) - 1)], 2) if vals else None,
        }
    output = {
        "ok": bool(samples),
        "model": engine.model_id,
        "device": str(cfg.get("device", "cpu")),
        "sidecar": engine.endpoint,
        "cold_load_ms": round(cold_ms, 2),
        "overall_median_ms": round(statistics.median(samples), 2) if samples else None,
        "overall_p95_ms": round(sorted(samples)[max(0, int(len(samples) * .95) - 1)], 2) if samples else None,
        "prompts": by_prompt,
    }
    if args.compare_fast:
        try:
            from ollama import Client
            agent_cfg = load_config().get("agent") or {}
            fast_model = str(agent_cfg.get("fast_model") or "agent-fast:2b")
            fast_options = dict(agent_cfg.get("fast_options") or {})
            fast_keep_alive = agent_cfg.get("fast_model_keep_alive", "2m")
            fast_client = Client(host=str(agent_cfg.get("host") or "http://127.0.0.1:11434"))
            fast_samples = []
            for prompt in PROMPTS:
                for _ in range(max(1, args.runs)):
                    result = route_with_fast_model(
                        fast_client, fast_model, prompt, options=fast_options, keep_alive=fast_keep_alive,
                    )
                    if result.ok:
                        fast_samples.append(result.latency_ms)
            output["fast_model_comparison"] = {
                "model": fast_model,
                "runs": len(fast_samples),
                "median_ms": round(statistics.median(fast_samples), 2) if fast_samples else None,
                "p95_ms": round(sorted(fast_samples)[max(0, int(len(fast_samples) * .95) - 1)], 2) if fast_samples else None,
                "median_ms_saved_by_laya": (
                    round(statistics.median(fast_samples) - statistics.median(samples), 2)
                    if fast_samples and samples else None
                ),
            }
        except Exception as exc:
            output["fast_model_comparison"] = {"error": str(exc)}
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
