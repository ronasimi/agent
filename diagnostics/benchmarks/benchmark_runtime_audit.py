#!/usr/bin/env python3
"""Repeatable offline routing/prompt benchmark; accepts an alternate source tree.

No inference or provider calls. Timing starts after imports/catalog load. Prompt
assembly includes history projection, stable policy, schema copy, telemetry and
admission control; it excludes SQLite hydration and turn-wide bookkeeping.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import time


def distribution(values):
    rows = sorted(values)
    return {"median": round(statistics.median(rows), 4), "p95": round(rows[math.ceil(len(rows) * .95) - 1], 4),
            "min": round(rows[0], 4), "max": round(rows[-1], 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from al_agent.deterministic_router import DeterministicToolRouter
    from al_agent.agent_loop import LoopConfig, fit_context, _prompt_telemetry
    from al_agent.prompts import build_system_prompt
    from al_agent.tool_session import ToolSession
    from tools.catalog import catalog_snapshot
    from tools.config import load_config
    from tools.state_tape import compact_recent_conversation

    config = load_config()["agent"]
    schemas, _, metadata = catalog_snapshot()
    router = DeterministicToolRouter()
    queries = ["Check the current time in UTC.", "What is the weather in London Ontario now?",
               "Calculate 19 times 23.", "List the connected devices on my local network.",
               "Read the current host CPU temperature and memory use.", "How many emails are in my inbox?"]
    raw_history = []
    for i in range(8):
        raw_history.extend([{"role": "user", "content": f"Check reading {i}.", "_db_id": i * 4 + 1},
                            {"role": "assistant", "content": "<tool_call>historical XML</tool_call>", "tool_calls": [{}]},
                            {"role": "tool", "content": json.dumps({"large_result": "x" * 5000})},
                            {"role": "assistant", "content": f"Verified reading {i}: 42.", "_db_id": i * 4 + 4}])
    route_ms, assembly_ms, inputs, schema_tokens, wire_bytes = [], [], [], [], []
    selections = {}
    loop_config = LoopConfig(model=config["model"], options=config["main_options"], reserve_tokens=2560,
                             soft_prompt_tokens=8192, hard_prompt_tokens=13824)
    for i in range(max(1, args.runs)):
        query = queries[i % len(queries)]
        started = time.perf_counter()
        decision = router.decide(query, schemas, metadata)
        route_ms.append((time.perf_counter() - started) * 1000)
        selections[query] = list(decision.selected)
        session = ToolSession(schemas, lambda *_: None, metadata, initial_active=list(decision.selected))
        started = time.perf_counter()
        active = copy.deepcopy(session.schemas)
        messages = [{"role": "system", "content": build_system_prompt()}]
        messages += compact_recent_conversation(raw_history, include_source_ids=True)
        messages.append({"role": "user", "content": query})
        _prompt_telemetry(messages, active)
        wire = fit_context(messages, active, loop_config)
        telemetry = _prompt_telemetry(wire, active)
        assembly_ms.append((time.perf_counter() - started) * 1000)
        inputs.append(telemetry["estimated_input_tokens"])
        schema_tokens.append(telemetry["estimated_schema_tokens"])
        wire_bytes.append(len(json.dumps([wire, active], ensure_ascii=False).encode()))
    print(json.dumps({"mode": "offline Python benchmark; no model or provider calls", "runs": len(route_ms),
                      "catalog_tools": len(schemas), "routing_ms": distribution(route_ms),
                      "prompt_assembly_ms": distribution(assembly_ms), "estimated_input_tokens": distribution(inputs),
                      "estimated_schema_tokens": distribution(schema_tokens), "wire_bytes": distribution(wire_bytes),
                      "selections": selections, "live_ollama_metrics": None}, indent=2))


if __name__ == "__main__":
    main()
