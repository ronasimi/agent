#!/usr/bin/env python3
"""Exercise real tool dispatch and error feedback with an offline scripted model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from al_agent.agent_loop import LoopConfig, run_loop
from al_agent.tool_session import ToolSession
from tools.catalog import catalog_snapshot
from tools.config import load_config
from tools.executor import execute_registered_tool


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        yield {
            "message": {"content": json.dumps(next(self.replies))},
            "done": True,
            "done_reason": "stop",
        }


def main():
    agent = load_config()["agent"]
    schemas, functions, metadata = catalog_snapshot()
    scenarios = {
        "direct_answer": [{"action": "final", "answer": "Hello"}],
        "discover_and_calculate": [
            {
                "action": "tool",
                "name": "load_tools",
                "arguments": {"names": ["calculate"]},
            },
            {
                "action": "tool",
                "name": "calculate",
                "arguments": {"expression": "19*23"},
            },
            {"action": "final", "answer": "437"},
        ],
        "repair_arguments": [
            {
                "action": "tool",
                "name": "load_tools",
                "arguments": {"names": ["calculate"]},
            },
            {"action": "tool", "name": "calculate", "arguments": {"wrong": "19*23"}},
            {
                "action": "tool",
                "name": "calculate",
                "arguments": {"expression": "19*23"},
            },
            {"action": "final", "answer": "437 after correcting the arguments"},
        ],
    }
    rows = []
    for name, replies in scenarios.items():
        client = ScriptedClient(replies)
        calls = []

        def execute(tool, arguments):
            calls.append(tool)
            return execute_registered_tool(
                tool, arguments, binding=(functions[tool], metadata[tool])
            )

        session = ToolSession(schemas, execute, metadata)
        answer = run_loop(
            [{"role": "system", "content": "Test"}, {"role": "user", "content": name}],
            client=client,
            tools=session,
            config=LoopConfig(agent["model"], agent["main_options"]),
            append=lambda message: None,
            emit=lambda *args, **kwargs: None,
        )
        rows.append(
            {
                "scenario": name,
                "model_calls": client.calls,
                "executed_tools": calls,
                "answer": answer,
            }
        )
    print(
        json.dumps(
            {
                "mode": "offline scripted transport; not model-quality evaluation",
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
