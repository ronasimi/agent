"""Opt-in live Ollama protocol checks.

Run with RUN_OLLAMA_LIVE_TESTS=1 pytest -q tests/test_ollama_conformance_live.py
on the target host. These tests are skipped in normal CI.
"""
from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_LIVE_TESTS") != "1",
    reason="requires RUN_OLLAMA_LIVE_TESTS=1 and a live configured Ollama server",
)


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def test_live_ollama_supports_harness_chat_metrics_and_tools():
    from ollama import Client
    from tools.config import load_config
    from tools.catalog import get_tool_schema

    cfg = load_config()["agent"]
    client = Client(host=cfg.get("host", "http://127.0.0.1:11434"))
    model = cfg["model"]
    options = {**(cfg.get("main_options") or {}), "num_predict": 4}
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": "Reply OK"}],
        tools=[get_tool_schema("current_time")],
        options=options,
        keep_alive=-1,
        think=False,
        stream=False,
    )
    assert _field(response, "prompt_eval_count", 0) is not None
    assert _field(response, "eval_count", 0) is not None
    assert _field(response, "message") is not None


def test_live_ollama_reasoning_recovery_mode_reaches_visible_content():
    """Catch reasoning-only completions that look successful at the API layer.

    The configured Qwen3.8-Distill roles may reason before answering. The
    bounded recovery deliberately disables thinking because this GGUF template
    only has a boolean thinking gate; string effort levels enter full reasoning.
    The recovery mode must reach ``message.content`` on the target Ollama build.
    """
    from ollama import Client
    from tools.config import load_config

    cfg = load_config()["agent"]
    recovery = dict(cfg.get("reasoning_recovery") or {})
    client = Client(host=cfg.get("host", "http://127.0.0.1:11434"))
    response = client.chat(
        model=cfg["model"],
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        options={
            **(cfg.get("main_options") or {}),
            "num_predict": int(recovery.get("final_num_predict", 2048)),
        },
        keep_alive=-1,
        think=False,
        stream=False,
    )
    message = _field(response, "message", {})
    content = str(_field(message, "content", "") or "").strip()
    thinking = str(_field(message, "thinking", "") or "").strip()
    assert content, (
        "Configured reasoning-recovery mode still returned no visible content "
        f"(thinking_chars={len(thinking)}, done_reason={_field(response, 'done_reason', 'unknown')!r})."
    )
