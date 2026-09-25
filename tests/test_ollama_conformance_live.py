"""Opt-in checks of the selected model/protocol on a real Ollama server."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_LIVE_TESTS") != "1",
    reason="requires RUN_OLLAMA_LIVE_TESTS=1 and a live configured Ollama server",
)


def _run(prompt):
    from ollama import Client
    from al_agent.agent_loop import LoopConfig, run_loop
    from al_agent.prompts import build_system_prompt
    from al_agent.tool_session import ToolSession
    from tools.catalog import catalog_snapshot
    from tools.config import load_config
    from tools.executor import execute_registered_tool

    cfg = load_config()["agent"]
    schemas, functions, metadata = catalog_snapshot()
    schemas = [
        s for s in schemas if s["function"]["name"] in {"calculate", "current_time"}
    ]
    calls = []

    def execute(name, arguments):
        calls.append(name)
        return execute_registered_tool(
            name, arguments, binding=(functions[name], metadata[name])
        )

    session = ToolSession(schemas, execute, metadata)
    answer = run_loop(
        [
            {
                "role": "system",
                "content": build_system_prompt() + "\n" + session.inventory(),
            },
            {"role": "user", "content": prompt},
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
    return answer, calls


def test_live_configured_protocol_produces_a_final_answer():
    answer, calls = _run("Reply with exactly OK. No tools are needed.")
    assert answer.strip() == "OK"
    assert not calls


def test_live_model_discovers_and_executes_calculate():
    answer, calls = _run(
        "Use the calculate tool to compute 19 * 23, then state the result."
    )
    assert "calculate" in calls
    assert "437" in answer


def test_live_model_discovers_and_executes_current_time():
    answer, calls = _run(
        "Use current_time to obtain the current UTC date and time, then report it."
    )
    assert "current_time" in calls
    assert answer.strip()
