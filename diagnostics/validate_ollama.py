#!/usr/bin/env python3
"""Live smoke test of the configured model and protocol with two read-only tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ollama import Client
from al_agent.agent_loop import LoopConfig, run_loop
from al_agent.prompts import build_system_prompt
from al_agent.tool_session import ToolSession
from tools.catalog import catalog_snapshot
from tools.config import load_config
from tools.executor import execute_registered_tool


def main():
    cfg = load_config()["agent"]
    schemas, functions, metadata = catalog_snapshot()
    schemas = [
        s for s in schemas if s["function"]["name"] in {"calculate", "current_time"}
    ]
    executed = []

    def execute(name, args):
        executed.append(name)
        return execute_registered_tool(
            name, args, binding=(functions[name], metadata[name])
        )

    session = ToolSession(schemas, execute, metadata)
    answer = run_loop(
        [
            {
                "role": "system",
                "content": build_system_prompt() + "\n" + session.inventory(),
            },
            {
                "role": "user",
                "content": "Use the calculate tool to compute 19 * 23, then tell me the result.",
            },
        ],
        client=Client(
            host=cfg["host"],
            timeout=cfg.get("model_transport", {}).get("timeout_seconds", 60),
        ),
        tools=session,
        config=LoopConfig(
            cfg["model"], cfg["main_options"], protocol=cfg.get("tool_protocol", "json")
        ),
        append=lambda message: None,
        emit=lambda *args, **kwargs: None,
        think_supported=bool(cfg.get("supports_thinking", False)),
    )
    if "calculate" not in executed or "437" not in answer:
        raise RuntimeError(
            f"Model smoke test failed: executed={executed!r}, answer={answer!r}"
        )
    print(
        json.dumps(
            {"ok": True, "model": cfg["model"], "tools": executed, "answer": answer},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
