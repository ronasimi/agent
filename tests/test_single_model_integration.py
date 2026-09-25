from __future__ import annotations

import inspect
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_autonomous_loop import FakeClient, final, tool


def test_all_legacy_roles_are_normalized_to_one_model(monkeypatch):
    from tools.config import normalize_config

    monkeypatch.delenv("AGENT_MODEL", raising=False)
    with pytest.warns(UserWarning, match="ignores legacy role overrides"):
        config = normalize_config(
            {
                "agent": {
                    "model": "distilled4b",
                    "fast_model": "tiny",
                    "vision_model": "vision",
                    "report_model": "large",
                    "main_options": {"num_ctx": 16384},
                    "compaction_options": {"num_ctx": 2048},
                    "semantic_memory_enabled": True,
                }
            }
        )
    a = config["agent"]
    for role in (
        "executor",
        "decision",
        "reasoning",
        "fast",
        "vision",
        "report",
        "compaction",
    ):
        assert a[role + "_model"] == "distilled4b"
        assert a[role + "_options"]["num_ctx"] == 16384
    assert a["semantic_memory_enabled"] is False


def test_no_production_embedding_calls_or_model_selection_modules():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for folder in ("al_agent", "tools", "webui"):
        for p in (root / folder).rglob("*.py"):
            assert ".embed(" not in p.read_text(), str(p)
    assert not (root / "al_agent/model_roles.py").exists()


def test_runtime_refreshes_saved_history_and_writes_complete_terminal_state(
    monkeypatch,
):
    from al_agent import runtime
    from al_agent import state
    from tools.conversation_context import conversation_context
    from tools.memory import _load_chat_history_from_db, apply_conversation_compaction
    from tools.working_state import WorkingStateStore

    cid = "integration-" + uuid.uuid4().hex
    client = FakeClient([final("First remembered answer"), final("Second answer")])
    monkeypatch.setattr(runtime, "OLLAMA", client)
    monkeypatch.setattr(state, "MODEL_TRACE_ENABLED", False)
    assert "refresh_history" in inspect.signature(runtime.handle_user_turn).parameters
    with conversation_context(cid):
        runtime.handle_user_turn(
            [], "Remember this first turn", False, refresh_history=True
        )
        first_history = _load_chat_history_from_db()
        assert apply_conversation_compaction(
            "Older turn summary", first_history[-1]["_db_id"]
        )
        runtime.handle_user_turn([], "What came before?", False, refresh_history=True)
        history = _load_chat_history_from_db()
        assert history[-1]["content"] == "Second answer"
        assert "First remembered answer" in json.dumps(client.requests[-1]["messages"])
        assert WorkingStateStore(conversation_id=cid).load()["status"] == "complete"


def test_runtime_executes_real_primitive_and_emits_ui_tool_names(monkeypatch):
    from al_agent import runtime, state
    from tools.conversation_context import conversation_context

    client = FakeClient(
        [
            tool("load_tools", names=["calculate"]),
            tool("calculate", expression="6 * 7"),
            final("42"),
        ]
    )
    monkeypatch.setattr(runtime, "OLLAMA", client)
    monkeypatch.setattr(state, "MODEL_TRACE_ENABLED", False)
    events = []
    with (
        conversation_context("primitive-" + uuid.uuid4().hex),
        runtime.frontend_event_context(events.append),
    ):
        runtime.handle_user_turn(
            [], "Calculate six times seven", False, refresh_history=True
        )
    results = [e for e in events if e["type"] == "tool_result"]
    assert [e["name"] for e in results] == ["load_tools", "calculate"]
    assert "42" in results[-1]["content"]
    assert events[-1]["type"] == "turn_end" and events[-1]["success"]


def test_foreground_tracking_does_not_clear_other_turns():
    from tools.runtime import set_foreground_turn, get_monitor_state

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: set_foreground_turn(f"test-{i}", {"pid": os.getpid()}),
                range(4),
            )
        )
    set_foreground_turn("test-0", None)
    active = get_monitor_state("agent.foreground_turns")
    assert all(f"test-{i}" in active for i in range(1, 4))
    for i in range(1, 4):
        set_foreground_turn(f"test-{i}", None)


def test_isolated_tool_inherits_conversation_context():
    from tools.executor import _execute_isolated_tool
    from tools.conversation_context import conversation_context
    from tools.goals import get_goal

    cid = "isolated-" + uuid.uuid4().hex
    with conversation_context(cid):
        _execute_isolated_tool("set_goal", {"goal": "only for this conversation"}, 10)
        assert "only for this conversation" in get_goal()
    with conversation_context("other-" + uuid.uuid4().hex):
        assert "only for this conversation" not in get_goal()


def test_semantic_compatibility_tools_do_not_call_another_model(monkeypatch):
    from tools import memory

    monkeypatch.setattr(memory, "remember", lambda **k: "stored lexically")
    monkeypatch.setattr(memory, "search_memory", lambda **k: "lexical result")
    assert memory.remember_semantic(fact="fact") == "stored lexically"
    assert memory.search_semantic_memory("fact") == "lexical result"


def test_background_and_foreground_share_model_and_context():
    from al_agent import state
    from al_agent.background import config
    from tools import deep_research, self_optimization, tool_manager

    for model in (
        config.MODEL,
        config.FAST_MODEL,
        config.REPORT_MODEL,
        config.COMPACTION_MODEL,
        deep_research.FAST_MODEL,
        self_optimization.MODEL,
        tool_manager.FAST_MODEL,
    ):
        assert model == state.MODEL
    for options in (
        config.FAST_OPTIONS,
        config.REPORT_OPTIONS,
        config.COMPACTION_OPTIONS,
        deep_research.FAST_OPTIONS,
        self_optimization.SELF_OPTIONS,
    ):
        assert options["num_ctx"] == state.MAX_CTX


def test_json_mode_history_contains_real_roles_not_protocol_bubbles(monkeypatch):
    from al_agent import runtime, state
    from tools.memory import _load_chat_history_from_db
    from tools.conversation_context import conversation_context

    monkeypatch.setattr(
        runtime,
        "OLLAMA",
        FakeClient(
            [
                tool("load_tools", names=["calculate"]),
                tool("calculate", expression="7*7"),
                final("49"),
            ]
        ),
    )
    monkeypatch.setattr(state, "MODEL_TRACE_ENABLED", False)
    with conversation_context("json-storage-" + uuid.uuid4().hex):
        runtime.handle_user_turn([], "Seven squared", False, refresh_history=True)
        history = _load_chat_history_from_db()
    assert len([m for m in history if m["role"] == "user"]) == 1
    assert len([m for m in history if m["role"] == "tool"]) == 2
    assert history[-1]["content"] == "49"


def test_websocket_preflight_failure_releases_run_slot():
    import asyncio
    from webui.chat import _run_turn, RUNS

    class Socket:
        async def send_json(self, payload):
            raise RuntimeError("disconnected before accepted")

    turn_id = uuid.uuid4().hex
    with pytest.raises(RuntimeError, match="disconnected"):
        asyncio.run(_run_turn(Socket(), {"turn_id": turn_id, "content": "hello"}))
    assert turn_id not in RUNS


def test_interrupted_native_history_gets_result_placeholders_in_order():
    from al_agent.turn_engine import _history_for_protocol

    history = [
        {"role": "user", "content": "do task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "function": {"name": "put", "arguments": {}}},
                {"id": "b", "function": {"name": "lookup", "arguments": {}}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "tool_name": "put", "content": "ok"},
    ]
    fixed = _history_for_protocol(history, "native")
    assert [m.get("tool_call_id") for m in fixed[2:]] == ["a", "b"]
    assert "unknown" in fixed[-1]["content"]


def test_history_matches_results_by_id_when_first_result_is_missing():
    from al_agent.turn_engine import _history_for_protocol

    history = [
        {"role": "tool", "tool_call_id": "orphan", "content": "orphan"},
        {"role": "user", "content": "do task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "function": {"name": "put", "arguments": {}}},
                {"id": "b", "function": {"name": "put", "arguments": {}}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "b",
            "tool_name": "put",
            "content": "second succeeded",
        },
        {"role": "user", "content": "continue"},
    ]
    fixed = _history_for_protocol(history, "native")
    assert fixed[2]["tool_call_id"] == "a" and "outcome_unknown" in fixed[2]["content"]
    assert fixed[3]["tool_call_id"] == "b" and fixed[3]["content"] == "second succeeded"
    assert fixed[4]["role"] == "user"


@pytest.mark.parametrize("protocol", ["json", "native", "qwen_xml"])
def test_real_ollama_sdk_serializes_actions_and_observations(protocol):
    import httpx
    from ollama import Client
    from al_agent.agent_loop import LoopConfig, run_loop
    from al_agent.tool_session import ToolSession
    from test_autonomous_loop import SCHEMA

    replies = iter(
        [tool("load_tools", names=["lookup"]), tool("lookup", key="x"), final("42")]
    )
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        action = next(replies)
        if protocol == "json":
            message = {"role": "assistant", "content": json.dumps(action)}
        elif protocol == "qwen_xml" and action["action"] == "tool":
            args = "".join(
                f"<parameter={key}>\n{json.dumps(value) if isinstance(value, (dict, list)) else value}\n</parameter>\n"
                for key, value in action["arguments"].items()
            )
            message = {
                "role": "assistant",
                "content": f"<tool_call>\n<function={action['name']}>\n{args}</function>\n</tool_call>",
            }
        elif action["action"] == "tool":
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": action["name"],
                            "arguments": action["arguments"],
                        }
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": action["answer"]}
        packet = {
            "model": "one-4b",
            "message": message,
            "done": True,
            "done_reason": "stop",
        }
        return httpx.Response(
            200,
            content=json.dumps(packet) + "\n",
            headers={"content-type": "application/x-ndjson"},
        )

    client = Client(
        host="http://ollama.test",
        transport=httpx.MockTransport(respond),
        trust_env=False,
    )
    executed = []
    session = ToolSession(
        [SCHEMA], lambda name, args: executed.append((name, args)) or {"value": 42}
    )
    answer = run_loop(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "lookup x"},
        ],
        client=client,
        tools=session,
        config=LoopConfig("one-4b", {"num_ctx": 16384}, protocol=protocol),
        append=lambda msg: None,
        emit=lambda *args, **kwargs: None,
    )
    assert answer == "42" and executed == [("lookup", {"key": "x"})]
    assert {r["model"] for r in requests} == {"one-4b"}
    assert "42" in json.dumps(requests[-1]["messages"])
    if protocol in {"native", "qwen_xml"}:
        assert requests[-1]["messages"][-1]["role"] == "user"
        assert requests[-1]["messages"][-1]["content"].startswith("<tool_response>\n")
        assert requests[-1]["messages"][-1]["content"].endswith("\n</tool_response>")
    else:
        assert requests[-1]["format"]["oneOf"]
