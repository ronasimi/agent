"""Regression tests for prompt-prefix stability and first-token latency work.

Each test pins one behavior that reduces Ollama prefill or model-load cost:
prompt-prefix reuse, a byte-stable tool set, model warm-up, and the bounded
control-note/grounding paths that otherwise grow the prompt every iteration.
"""
from __future__ import annotations

from tools.context import build_active_messages


def _history():
    return [
        {"role": "user", "content": "STABLE EARLIER REQUEST"},
        {"role": "assistant", "content": "STABLE EARLIER ANSWER"},
        {"role": "user", "content": "CURRENT REQUEST"},
    ]


def _render(messages):
    return "\n".join(f"{m.get('role')}:{m.get('content', '')}" for m in messages)


def test_working_state_is_merged_into_single_leading_system_message():
    result = build_active_messages(
        system_prompt="stable-system",
        summary="",
        history=_history(),
        max_ctx_tokens=8000,
        reserve_tokens=500,
        working_state='{"objective":"CURRENT REQUEST","evidence":["a"]}',
        evidence_context="observed evidence",
    )
    roles = [m["role"] for m in result]
    assert roles[0] == "system"
    assert [i for i, role in enumerate(roles) if role == "system"] == [0]
    contents = [str(m.get("content") or "") for m in result]
    history_index = next(i for i, c in enumerate(contents) if "CURRENT REQUEST" == c)
    state_index = next(i for i, c in enumerate(contents) if "Harness working state" in c)
    evidence_index = next(i for i, c in enumerate(contents) if "Private untrusted evidence" in c)
    assert state_index == 0
    assert evidence_index == 0
    assert history_index > 0
    assert all("Harness evidence digest" not in c for c in contents)
    assert "stable-system" in result[0]["content"]


def test_changed_working_state_preserves_non_system_history_order():
    """Dynamic state may invalidate KV prefix but must never alter chat chronology."""
    first = build_active_messages(
        system_prompt="stable-system",
        summary="",
        history=_history(),
        max_ctx_tokens=8000,
        reserve_tokens=500,
        working_state='{"step":1}',
    )
    second = build_active_messages(
        system_prompt="stable-system",
        summary="",
        history=_history(),
        max_ctx_tokens=8000,
        reserve_tokens=500,
        working_state='{"step":2,"evidence":["new observation"]}',
    )
    assert first[0]["role"] == second[0]["role"] == "system"
    assert first[0]["content"] != second[0]["content"]
    assert first[1:] == second[1:]
    assert all(message["role"] != "system" for message in first[1:])


def test_volatile_last_flag_cannot_create_a_late_system_message():
    result = build_active_messages(
        system_prompt="stable-system",
        summary="",
        history=_history(),
        max_ctx_tokens=8000,
        reserve_tokens=500,
        working_state='{"objective":"CURRENT REQUEST"}',
        volatile_last=False,
    )
    assert [i for i, message in enumerate(result) if message["role"] == "system"] == [0]
    assert "Harness working state" in result[0]["content"]


def test_volatile_blocks_are_counted_against_the_token_budget():
    """Placing the blocks last must not let them escape the context budget."""
    result = build_active_messages(
        system_prompt="system",
        summary="",
        history=[{"role": "user", "content": "x " * 400}],
        max_ctx_tokens=900,
        reserve_tokens=100,
        working_state="w " * 400,
    )
    from tools.context import estimate_messages_tokens

    assert estimate_messages_tokens(result) <= 800












def test_network_status_phrasing_creates_a_network_requirement():
    from tools.task_requirements import TaskRequirementLedger

    ledger = TaskRequirementLedger.from_request(
        "Check host memory, disk usage, network status and the git repo status."
    )
    assert "network_snapshot" in ledger.required_tools()


def test_warm_model_loads_weights_and_primes_the_system_prefix():
    from al_agent.model_protocol import warm_model

    calls = []

    class Client:
        def chat(self, **kwargs):
            calls.append(kwargs)
            return {"message": {"content": ""}}

    assert warm_model(
        Client(), "main:4b", options={"num_ctx": 16384, "temperature": 0.2},
        keep_alive=-1, system_prompt="stable-system",
        tools=[{"type": "function", "function": {"name": "tool_search", "parameters": {"type": "object"}}}],
        think=False,
    ) is True
    assert len(calls) == 1
    assert calls[0]["keep_alive"] == -1
    # The runner is keyed by context size: warming with a different num_ctx
    # would reload the model on the first real turn.
    assert calls[0]["options"]["num_ctx"] == 16384
    assert calls[0]["options"]["num_predict"] == 1
    assert calls[0]["messages"][0]["role"] == "system"
    assert calls[0]["messages"][1]["role"] == "user"
    assert calls[0]["think"] is False
    assert calls[0]["tools"][0]["function"]["name"] == "tool_search"


def test_warm_model_failure_is_reported_and_never_raises():
    from al_agent.model_protocol import warm_model

    errors = []

    class Client:
        def chat(self, **kwargs):
            raise ConnectionError("ollama is not running")

    assert warm_model(Client(), "main:4b", on_error=errors.append) is False
    assert errors and isinstance(errors[0], ConnectionError)


def test_compaction_reuses_the_interactive_context_size_for_the_same_model():
    """A smaller num_ctx on the same model would unload the warm interactive runner."""
    from al_agent.background import config as background_config

    if background_config.COMPACTION_MODEL == background_config.MODEL:
        assert background_config.COMPACTION_OPTIONS["num_ctx"] == background_config.MAIN_OPTIONS["num_ctx"]






class _ScriptedOllama:
    """Minimal streaming Ollama stand-in for turn-level regression tests."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        if not kwargs.get("stream"):
            return {"message": {"content": self.reply(len(self.requests), kwargs)}, "done": True}
        return self._stream(self.reply(len(self.requests), kwargs))

    def _stream(self, content):
        yield {"message": {"content": str(content)}, "done": False}
        yield {"message": {"content": ""}, "done": True, "prompt_eval_count": 10, "eval_count": 5}


def _run_turn(monkeypatch, prompt, reply):
    from al_agent import runtime as agent
    from al_agent import events, turn_support

    client = _ScriptedOllama(reply)
    monkeypatch.setattr(agent, "OLLAMA", client)
    monkeypatch.setattr(agent, "LOOP_VALIDATOR_CLIENT", client)
    monkeypatch.setattr(turn_support, "OLLAMA", client)
    captured = []
    messages = [{"role": "system", "content": "system"}]
    with events.frontend_event_context(sink=captured.append):
        agent.handle_user_turn(messages, prompt, False)
    return client, captured
