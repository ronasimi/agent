"""Behavioral regression tests for the replacement action/result architecture."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

from al_agent.agent_loop import (
    LoopConfig,
    LoopStopped,
    decode_response,
    fit_context,
    run_loop,
)
from al_agent.tool_session import ToolSession, validate_arguments

SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Read a named value",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}
WRITE = {
    "type": "function",
    "function": {
        "name": "put",
        "description": "Write a named value",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "integer"}},
            "required": ["key", "value"],
            "additionalProperties": False,
        },
    },
}
CFG = LoopConfig(
    model="original-distilled-4b", options={"num_ctx": 16384, "num_predict": 2048},
    protocol="json",
)


def tool(name, **arguments):
    return {"action": "tool", "name": name, "arguments": arguments}


def final(text="Done"):
    return {"action": "final", "answer": text}


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.closed = 0

    def chat(self, **request):
        self.requests.append(copy.deepcopy(request))
        reply = next(self.replies)

        def stream():
            try:
                if isinstance(reply, Exception):
                    raise reply
                if isinstance(reply, list):
                    yield from reply
                    return
                content = reply if isinstance(reply, str) else json.dumps(reply)
                yield {"message": {"content": content[:12]}, "done": False}
                yield {
                    "message": {"content": content[12:]},
                    "done": True,
                    "done_reason": "stop",
                }
            finally:
                self.closed += 1

        return stream()


def run(replies, *, execute=None, config=CFG, cancel=lambda: False, slot=None):
    calls, saved, events = [], [], []

    def execute_fn(name, args):
        calls.append((name, args))
        return execute(name, args) if execute else {"value": 42}

    session = ToolSession(
        [SCHEMA, WRITE],
        execute_fn,
        {"lookup": {"readonly": True}, "put": {"readonly": False}},
    )
    client = FakeClient(replies)
    args = dict(
        client=client,
        tools=session,
        config=config,
        append=saved.append,
        emit=lambda kind, **data: events.append((kind, data)),
        cancel=cancel,
        store_observation=lambda name, text: "obs-1",
    )
    if slot:
        args["inference_slot"] = slot
    result = run_loop(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "request"},
        ],
        **args,
    )
    return result, calls, saved, events, client, session


@pytest.mark.parametrize(
    "prompt",
    [
        "Check the weather, email, LAN and a document",
        'Explain the example "delete a file", without executing it',
        "What time is it?",
        "Hello",
        "Refactor this program",
    ],
)
def test_prompt_never_forces_execution(prompt):
    client = FakeClient([final("A model-chosen answer")])
    calls = []
    session = ToolSession([SCHEMA], lambda *args: calls.append(args))
    result = run_loop(
        [{"role": "system", "content": "policy"}, {"role": "user", "content": prompt}],
        client=client,
        tools=session,
        config=CFG,
        append=lambda m: None,
        emit=lambda *a, **k: None,
    )
    assert result == "A model-chosen answer" and calls == []
    assert len(client.requests) == 1


def test_model_selects_dependent_calls_and_receives_results():
    result, calls, saved, events, client, _ = run(
        [
            tool("load_tools", names=["lookup", "put"]),
            tool("lookup", key="source"),
            tool("put", key="destination", value=42),
            final("Stored 42"),
        ]
    )
    assert result == "Stored 42"
    assert calls == [
        ("lookup", {"key": "source"}),
        ("put", {"key": "destination", "value": 42}),
    ]
    assert "42" in json.dumps(client.requests[-1]["messages"])
    assert all(r["model"] == "original-distilled-4b" for r in client.requests)
    assert client.closed == 4
    assert len([e for e in events if e[0] == "assistant_final"]) == 1


def test_invalid_calls_are_returned_for_model_correction():
    result, calls, *_ = run(
        [
            tool("put", key="x", value=1),
            tool("load_tools", names=["put"]),
            tool("put", key="x", value="invalid"),
            tool("put", key="x", value=1),
            final(),
        ]
    )
    assert calls == [("put", {"key": "x", "value": 1})]


def test_model_can_discover_without_known_keyword_mapping():
    result, calls, *_ = run(
        [tool("tool_search", query="named value"), tool("lookup", key="x"), final()]
    )
    assert calls == [("lookup", {"key": "x"})]


def test_no_native_schema_or_think_parameter_in_default_json_mode():
    *_, client, session = run([final()])
    assert "format" in client.requests[0] and "tools" not in client.requests[0]
    assert "think" not in client.requests[0]


def test_qwen_xml_mode_uses_tools_parses_xml_and_injects_exact_tool_response():
    config = replace(CFG, protocol="qwen_xml")
    xml_load = """<tool_call>\n<function=load_tools>\n<parameter=names>\n[\"put\"]\n</parameter>\n</function>\n</tool_call>"""
    xml_put = """<tool_call>\n<function=put>\n<parameter=key>\nx\n</parameter>\n<parameter=value>\n7\n</parameter>\n</function>\n</tool_call>"""
    result, calls, _saved, _events, client, _session = run(
        [xml_load, xml_put, "Done"], config=config
    )
    assert result == "Done"
    assert calls == [("put", {"key": "x", "value": 7})]
    assert "tools" in client.requests[0] and "format" not in client.requests[0]
    assert any(
        message.get("role") == "user"
        and message.get("content", "").startswith("<tool_response>\n")
        and message.get("content", "").endswith("\n</tool_response>")
        for message in client.requests[-1]["messages"]
    )


def test_qwen_xml_no_think_request_is_explicit_when_supported():
    config = replace(CFG, protocol="qwen_xml")
    session = ToolSession([], lambda *args: None)
    client = FakeClient(["OK"])
    run_loop(
        [{"role": "system", "content": "policy"}, {"role": "user", "content": "hello"}],
        client=client,
        tools=session,
        config=config,
        append=lambda _m: None,
        emit=lambda *_a, **_k: None,
        thinking=False,
        think_supported=True,
    )
    assert client.requests[0]["think"] is False


def test_oversized_schema_activation_is_atomic_and_evicts_only_older_tools():
    large = copy.deepcopy(WRITE)
    large["function"]["description"] = "x" * 2100
    session = ToolSession([SCHEMA, large], lambda *args: None, max_schema_chars=2000)
    session.invoke("load_tools", {"names": ["lookup"]})
    with pytest.raises(ValueError, match="context allowance"):
        session.invoke("load_tools", {"names": ["put"]})
    assert list(session.active) == ["lookup"]
    session.max_schema_chars = 2400
    result = session.invoke("load_tools", {"names": ["put"]})
    assert result["evicted"] == ["lookup"]
    assert list(session.active) == ["put"]


@pytest.mark.parametrize(
    "arguments", [{"key": 3}, {"key": "a", "unknown": 1}, {}, [], None]
)
def test_argument_schema_is_strict(arguments):
    with pytest.raises((ValueError, TypeError)):
        validate_arguments(SCHEMA, arguments)


def test_rejects_non_finite_numbers_and_duplicate_json_keys():
    with pytest.raises(ValueError):
        validate_arguments(WRITE, {"key": "x", "value": float("nan")})
    with pytest.raises(ValueError):
        decode_response('{"action":"final","answer":"a","answer":"b"}', [], "json")


def test_never_executes_prose_or_xml():
    with pytest.raises(LoopStopped):
        run(
            [
                "<tool_call><function=put><parameter=key>x</parameter></function></tool_call>"
            ]
            * 3
        )


def test_thinking_only_output_is_not_promoted_to_answer():
    replies = [[{"message": {"thinking": "private draft"}, "done": True}]] * 3
    with pytest.raises(LoopStopped, match="valid action"):
        run(replies)


def test_incomplete_stream_does_not_execute_even_complete_action_text():
    with pytest.raises(LoopStopped, match="completion marker"):
        run(
            [
                [
                    {
                        "message": {
                            "content": json.dumps(tool("load_tools", names=["put"]))
                        },
                        "done": False,
                    }
                ]
            ]
        )


def test_stream_exception_is_not_replayed():
    client = FakeClient([ConnectionError("disconnected"), final()])
    with pytest.raises(LoopStopped, match="disconnected"):
        run_loop(
            [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "go"},
            ],
            client=client,
            tools=ToolSession([], lambda *a: None),
            config=CFG,
            append=lambda m: None,
            emit=lambda *a, **k: None,
        )
    assert len(client.requests) == 1 and client.closed == 1


def test_cancel_before_first_call_and_budget_exhaustion():
    with pytest.raises(LoopStopped, match="cancelled"):
        run([final()], cancel=lambda: True)
    with pytest.raises(LoopStopped, match="model-call budget"):
        run(
            [tool("load_tools", names=["lookup"])],
            config=replace(CFG, max_model_calls=1),
        )


def test_failure_can_be_recovered_by_same_model_without_forced_fallback():
    result, calls, *_ = run(
        [
            tool("load_tools", names=["lookup"]),
            tool("lookup", key="bad"),
            tool("lookup", key="good"),
            final("Recovered"),
        ],
        execute=lambda name, args: (
            {"ok": False, "error": "unknown key"}
            if args["key"] == "bad"
            else {"ok": True, "value": 42}
        ),
    )
    assert result == "Recovered" and len(calls) == 2


def test_unknown_mutation_outcome_is_not_replayed_in_same_turn():
    calls = []

    def fail(name, args):
        calls.append(name)
        raise TimeoutError("may have committed")

    session = ToolSession([WRITE], fail, {"put": {"readonly": False}})
    session.invoke("load_tools", {"names": ["put"]})
    with pytest.raises(TimeoutError):
        session.invoke("put", {"key": "x", "value": 1})
    second = session.invoke("put", {"key": "x", "value": 1})
    assert second["outcome_unknown"] and calls == ["put"]


def test_inference_lock_is_released_before_tools_run():
    held = []

    @contextmanager
    def slot():
        held.append(True)
        try:
            yield
        finally:
            held.pop()

    def execute(*args):
        assert not held
        return "ok"

    run(
        [tool("load_tools", names=["lookup"]), tool("lookup", key="x"), final()],
        execute=execute,
        slot=slot,
    )
    assert not held


def test_context_never_drops_current_request_to_keep_synthetic_tool_results():
    config = replace(CFG, options={"num_ctx": 3000}, reserve_tokens=1000)
    original = {"role": "user", "content": "original task " + ("x" * 2000)}
    data = [
        {"role": "system", "content": "policy"},
        original,
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "tool data " * 1000, "_runtime": True},
    ]
    with pytest.raises(LoopStopped, match="context budget"):
        fit_context(data, [], config)
    assert data[1] == original


def test_context_evicts_whole_old_turns():
    config = replace(CFG, options={"num_ctx": 3000}, reserve_tokens=1000)
    data = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "x" * 15000},
        {"role": "user", "content": "current task"},
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "result", "_runtime": True},
    ]
    fitted = fit_context(data, [], config)
    assert fitted[1]["content"] == "current task"
    assert len(fitted) == 4


def test_native_calls_have_matching_results_and_no_inferred_calls():
    def native(*calls):
        return [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": n, "arguments": a}} for n, a in calls
                    ],
                },
                "done": True,
            }
        ]

    result, calls, saved, events, client, _ = run(
        [
            native(("load_tools", {"names": ["lookup"]})),
            native(("lookup", {"key": "a"}), ("lookup", {"key": "b"})),
            [{"message": {"content": "Two results"}, "done": True}],
        ],
        config=replace(CFG, protocol="native"),
    )
    assert len(calls) == 2 and result == "Two results"
    assert len([m for m in saved if m["role"] == "tool"]) == 3
    assert all("format" not in r for r in client.requests)
    for r in client.requests:
        assert [i for i, m in enumerate(r["messages"]) if m["role"] == "system"] == [0]


def test_native_batch_rejects_all_before_any_invalid_side_effect():
    reply = [
        {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "put",
                            "arguments": {"key": "x", "value": 1},
                        }
                    },
                    {
                        "function": {
                            "name": "put",
                            "arguments": {"key": "x", "value": "bad"},
                        }
                    },
                ]
            },
            "done": True,
        }
    ]
    _, calls, *_ = run(
        [
            [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "load_tools",
                                    "arguments": {"names": ["put"]},
                                }
                            }
                        ]
                    },
                    "done": True,
                }
            ],
            reply,
            [{"message": {"content": "Could not call"}, "done": True}],
        ],
        config=replace(CFG, protocol="native"),
    )
    assert calls == []


def test_tool_session_state_is_not_shared_and_full_catalog_is_discoverable():
    one = ToolSession([SCHEMA], lambda *a: None)
    two = ToolSession([SCHEMA], lambda *a: None)
    one.invoke("load_tools", {"names": ["lookup"]})
    assert "lookup" in one.active and "lookup" not in two.active
    result = two.invoke("tool_search", {})
    assert result["total"] == 1 and result["selected"] == "lookup"
    assert result["activated"] == ["lookup"]
    assert "schemas" not in result


def test_native_stream_preserves_distinct_identical_argument_calls():
    from al_agent.model_protocol import merge_stream_tool_calls

    a = {"function": {"index": 0, "name": "lookup", "arguments": {"key": "same"}}}
    b = {"function": {"index": 1, "name": "lookup", "arguments": {"key": "same"}}}
    calls = merge_stream_tool_calls([], [a, b])
    assert len(calls) == 2
    assert len(merge_stream_tool_calls(calls, [a])) == 2


def test_prompt_telemetry_is_attached_to_model_trace_request():
    traced = []
    session = ToolSession([SCHEMA], lambda *_: {"value": 1})
    client = FakeClient([final("done")])
    run_loop(
        [{"role": "system", "content": "policy"}, {"role": "user", "content": "hello"}],
        client=client,
        tools=session,
        config=CFG,
        append=lambda _: None,
        emit=lambda *args, **kwargs: None,
        trace=lambda **kwargs: traced.append(kwargs),
    )
    telemetry = traced[-1]["request"]["prompt_telemetry"]
    assert telemetry["message_count"] == 2
    assert telemetry["estimated_input_tokens"] > 0
    assert telemetry["schema_chars"] > 0
    assert telemetry["historical_tool_messages"] == 0
