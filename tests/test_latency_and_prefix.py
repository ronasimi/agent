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


def test_unchanged_schema_set_is_left_byte_stable():
    """Pruning or reordering alone would invalidate the whole prompt cache."""
    from al_agent.runtime import _refresh_requirement_tool_schemas
    from tools import TOOL_METADATA, get_tool_schema
    from tools.task_requirements import TaskRequirementLedger
    from tools.turn_policy import derive_turn_tool_policy

    request = "Inspect host CPU and memory and CPU memory IO pressure"
    ledger = TaskRequirementLedger.from_request(request)
    policy = derive_turn_tool_policy(request, set(TOOL_METADATA), TOOL_METADATA)
    schemas = [get_tool_schema("host_snapshot"), get_tool_schema("pressure_snapshot")]
    ledger.record_tool("host_snapshot", status="ok", fingerprint="host-one")

    changed = _refresh_requirement_tool_schemas(
        schemas, ledger, policy, minimize_churn=True
    )
    names = [schema["function"]["name"] for schema in schemas]
    assert changed is False
    # Neither removed nor reordered: the serialized set stays identical.
    assert names == ["host_snapshot", "pressure_snapshot"]


def test_satisfied_schema_is_pruned_when_the_set_changes_anyway():
    from al_agent.runtime import _refresh_requirement_tool_schemas
    from tools import TOOL_METADATA, get_tool_schema
    from tools.task_requirements import TaskRequirementLedger
    from tools.turn_policy import derive_turn_tool_policy

    request = "Inspect host CPU and memory and CPU memory IO pressure"
    ledger = TaskRequirementLedger.from_request(request)
    policy = derive_turn_tool_policy(request, set(TOOL_METADATA), TOOL_METADATA)
    # pressure_snapshot is still pending but absent, so this iteration must
    # change the tool set regardless; pruning is then free.
    schemas = [get_tool_schema("host_snapshot")]
    ledger.record_tool("host_snapshot", status="ok", fingerprint="host-one")

    changed = _refresh_requirement_tool_schemas(
        schemas, ledger, policy, minimize_churn=True
    )
    names = [schema["function"]["name"] for schema in schemas]
    assert changed is True
    assert "host_snapshot" not in names
    assert "pressure_snapshot" in names


def test_schema_selection_ignores_single_incidental_description_matches():
    from tools import select_tool_schemas

    names = {
        schema["function"]["name"]
        for schema in select_tool_schemas("Explain in two sentences what an agent harness does.", max_tools=12)
    }
    assert names == set()


def test_generic_execution_is_not_exposed_by_a_vague_lexical_match():
    from tools import select_tool_schemas

    vague = {
        schema["function"]["name"]
        for schema in select_tool_schemas("tell me about this agent", max_tools=12)
    }
    assert "execute_shell" not in vague
    assert "execute_python" not in vague

    named = {
        schema["function"]["name"]
        for schema in select_tool_schemas("run a shell command to list /etc", max_tools=12)
    }
    assert "execute_shell" in named


def test_relevant_diagnostic_selection_is_unchanged():
    from tools import select_tool_schemas

    names = {
        schema["function"]["name"]
        for schema in select_tool_schemas("Check host memory and disk usage", max_tools=12)
    }
    assert {"host_snapshot", "memory_info", "disk_usage"} <= names


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
    ) is True
    assert calls[0]["messages"] == []
    assert calls[0]["keep_alive"] == -1
    # The runner is keyed by context size: warming with a different num_ctx
    # would reload the model on the first real turn.
    assert calls[0]["options"]["num_ctx"] == 16384
    assert calls[1]["options"]["num_ctx"] == 16384
    assert calls[1]["options"]["num_predict"] == 1
    assert calls[1]["messages"][0]["role"] == "system"


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


def test_finalization_strips_local_only_fields_from_the_wire():
    from al_agent import runtime as agent

    captured = {}

    class FakeClient:
        def chat(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            captured["stream"] = kwargs.get("stream")
            return {"message": {"content": "final"}}

    original_client = agent.OLLAMA
    original_save = agent.append_and_save
    agent.OLLAMA = FakeClient()
    agent.append_and_save = lambda messages, message: messages.append(message)
    try:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "inspect"},
        ]
        tail = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "demo", "arguments": {}}}]},
            {"role": "tool", "content": "observed", "tool_name": "demo", "tool_call_id": "c1"},
        ]
        agent._finalize_after_limit(messages, tail)
    finally:
        agent.OLLAMA = original_client
        agent.append_and_save = original_save

    assert captured["stream"] is True
    assert all("tool_call_id" not in message for message in captured["messages"])
    assert [i for i, message in enumerate(captured["messages"]) if message.get("role") == "system"] == [0]
    assert messages[-1]["content"] == "final"


def test_config_exposes_the_latency_knobs():
    import yaml

    with open("config/config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    agent_cfg = config["agent"]
    assert agent_cfg["warmup"]["enabled"] is True
    assert agent_cfg["context"]["volatile_blocks_last"] is False
    assert agent_cfg["working_state"]["minimize_schema_churn"] is True
    assert int(agent_cfg["grounding"]["max_candidate_discards"]) >= 1


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


def test_grounding_gate_stops_discarding_after_its_budget(monkeypatch, capsys):
    """Without a budget this path silently consumes the whole iteration limit."""
    from al_agent import runtime as agent

    client, events_seen = _run_turn(
        monkeypatch,
        "What is the weather in London Ontario right now?",
        lambda index, request: "It is sunny and 20C.",
    )
    capsys.readouterr()
    finals = [e for e in events_seen if e["type"] == "assistant_final"]
    assert finals and finals[-1].get("grounded") is False
    # One inference per discard plus the recovery attempt, never the full budget.
    assert len(client.requests) <= agent.GROUNDING_MAX_DISCARDS + 2


def test_repeated_harness_control_notes_are_not_duplicated(monkeypatch, capsys):
    client, _events = _run_turn(
        monkeypatch,
        "What is the weather in London Ontario right now?",
        lambda index, request: "It is sunny and 20C.",
    )
    capsys.readouterr()
    last = client.requests[-1]["messages"]
    gate_notes = [
        message for message in last
        if message.get("role") == "user"
        and str(message.get("content") or "").startswith("[Harness hard grounding gate]")
    ]
    assert len(gate_notes) <= 1


def test_simple_chat_does_not_expose_private_harness_context(monkeypatch, capsys):
    """Greetings must use the minimal conversational path, not narrate harness state."""
    client, _events = _run_turn(
        monkeypatch,
        "Hi, how are you?",
        lambda index, request: "Hi! I'm doing well, thanks for asking.",
    )
    capsys.readouterr()
    assert len(client.requests) == 1
    wire = "\n".join(str(message.get("content") or "") for message in client.requests[0]["messages"])
    for private_marker in (
        "Harness Evidence Digest",
        "Harness working state",
        "Harness scheduler",
        "Harness recipe preflight",
        "Private untrusted evidence",
        "validator_history",
    ):
        assert private_marker not in wire
