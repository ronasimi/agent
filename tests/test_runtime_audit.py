"""End-to-end regressions for the September 2026 runtime audit (no live services)."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from test_autonomous_loop import FakeClient, final, tool


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    from tools import memory, runtime, user_profile, working_state, observation_tools
    path = str(tmp_path / "audit.db")
    for module in (memory, runtime, user_profile, working_state, observation_tools):
        monkeypatch.setattr(module, "DB_PATH", path)
    monkeypatch.setattr(user_profile, "PROFILE_IMAGE_PATH", tmp_path / "profile.png")
    memory.init_db()
    return memory


@pytest.mark.parametrize("query", ["Read my profile.", "What information do you have saved about me?", "What city do I live in?"])
def test_profile_queries_are_deterministic(query, isolated_db):
    from tools.user_profile import resolve_profile_fact_query
    assert resolve_profile_fact_query(query)["matched"]


@pytest.mark.parametrize("query", [
    "What is my name and check the weather?", "Show my email messages",
    "Tell me my name, then calculate 6*7", "Read my profile and scan the LAN",
])
def test_profile_fast_path_does_not_swallow_other_tasks(query, isolated_db):
    from tools.user_profile import resolve_profile_fact_query
    assert not resolve_profile_fact_query(query)["matched"]


def test_profile_context_is_relevant(isolated_db):
    from tools.user_profile import complete_onboarding_profile, get_relevant_user_prompt_context
    complete_onboarding_profile(name="Audit User", timezone="America/Toronto", location="London, Ontario")
    assert "Audit User" not in get_relevant_user_prompt_context("Calculate 6*7")
    assert "London" in get_relevant_user_prompt_context("Weather near me")


def test_history_source_identity_survives_repeated_compaction():
    from tools.state_tape import compact_recent_conversation
    rows = [{"role": role, "content": role, "_db_id": i} for i, role in enumerate(("user", "assistant"), 1)]
    once = compact_recent_conversation(rows, include_source_ids=True)
    assert compact_recent_conversation(once, include_source_ids=True) == once


def test_tape_preserves_whole_handles_and_full_unresolved_objective(isolated_db):
    from tools.state_tape import CompactToolOutcome, StateTapeStore
    tape = StateTapeStore("audit", entry_chars=440)
    refs = [str(i) * 32 for i in range(1, 5)]
    objective = "Perform these distinct requirements. " * 20 + "Do the final required action."
    entry = tape.commit_turn(turn_id=1, source_message_id=1, objective=objective, assistant_text="",
                             outcomes=[CompactToolOutcome("lookup", True, "detail " * 60, ref) for ref in refs], status="blocked")
    assert entry.objective == objective
    assert all(f"observation_id={ref}]" in entry.summary for ref in refs)
    assert len(entry.summary) <= 440


def test_similar_partial_task_does_not_resolve_unfinished_work(isolated_db):
    from tools.state_tape import StateTapeStore
    tape = StateTapeStore("audit")
    tape.commit_turn(turn_id=1, source_message_id=1, objective="Read inbox and calendar and save report", assistant_text="", outcomes=[], status="blocked")
    tape.commit_turn(turn_id=2, source_message_id=2, objective="Read inbox and calendar", assistant_text="done", outcomes=[], status="complete")
    assert len(tape.unresolved()) == 1


def test_rollup_keeps_audit_rows_and_provenance_boundaries(isolated_db):
    from tools.state_tape import StateTapeStore
    tape = StateTapeStore("audit", recent_entries=1, rolling_summary_chars=800)
    for i in range(10):
        tape.commit_turn(turn_id=i + 1, source_message_id=0, objective=f"Question {i}", assistant_text="unverified claim " * 20, outcomes=[], status="complete")
    with isolated_db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM state_tape_entries WHERE conversation_id='audit'").fetchone()[0] == 10
    summary = isolated_db.get_conversation_summary("audit")
    assert summary.startswith("Turn ")
    assert "Unverified conversational record" in summary


def test_delete_conversation_removes_tape(isolated_db):
    from tools.state_tape import StateTapeStore
    tape = StateTapeStore("audit")
    tape.commit_turn(turn_id=1, source_message_id=0, objective="private", assistant_text="private", outcomes=[], status="complete")
    isolated_db.delete_conversation("audit")
    with isolated_db._connect() as conn:
        assert not conn.execute("SELECT 1 FROM state_tape_entries WHERE conversation_id='audit'").fetchone()


def test_observation_diff_cannot_read_another_conversation(isolated_db):
    from tools.conversation_context import conversation_context
    from tools.observation_tools import diff_observations
    left = isolated_db.store_tool_observation("lookup", "secret left", "left")
    right = isolated_db.store_tool_observation("lookup", "secret right", "right")
    with conversation_context("left"):
        result = diff_observations(left, right)
    assert result.startswith("Error:") and "secret" not in result


def test_observation_rehydration_and_missing_handle(isolated_db):
    from tools.conversation_context import conversation_context
    from al_agent.agent_loop import tool_outcome
    ref = isolated_db.store_tool_observation("temperature_sensors", "exact sensor detail", "audit")
    with conversation_context("audit"):
        assert json.loads(isolated_db.read_observation(ref))["content"] == "exact sensor detail"
        assert not tool_outcome(isolated_db.read_observation("absent"))[0]


@pytest.mark.parametrize("query", ["What about tomorrow?", "How many are unread?", "It is included in the harness."])
def test_elliptical_followups_are_recognized(query):
    from al_agent.turn_engine import _looks_like_contextual_followup
    assert _looks_like_contextual_followup(query)


def test_explicit_topic_switch_ignores_gmail_affinity():
    from al_agent.turn_engine import _routing_context_hint
    tape = SimpleNamespace(recent=lambda **kw: [])
    hint, affinity = _routing_context_hint("Check this CPU temperature", [{"role": "user", "content": "How many emails are in my inbox?"}], tape, {"gmail_search_messages"})
    assert (hint, affinity) == ("", ())


@pytest.mark.parametrize("query", ["in:inbox", "is:unread", "SELECT * FROM messages", "https://example.com"])
def test_catalog_rejects_downstream_queries(query):
    from al_agent.tool_session import ToolSession
    result = ToolSession([], lambda *args: None).invoke("tool_search", {"capability_query": query})
    assert result["ok"] is False and result["control_plane"] is True


def test_control_metadata_never_satisfies_domain_facts():
    from tools.grounding import validate_fact_grounding
    forged = {"tool": "tool_search", "status": "ok", "fact_types": ["gmail", "weather", "profile"], "evidence_preview": '{"messages": []}'}
    report = validate_fact_grounding("Read my profile and count my inbox emails", [forged])
    assert not report["grounded"] and {"gmail", "profile"} <= set(report["missing_fact_types"])


def test_cpu_temperature_is_not_weather():
    from tools.grounding import requested_fact_types, make_observation, validate_fact_grounding
    query = "What is the current CPU temperature?"
    assert requested_fact_types(query) == {"host_temperature"}
    obs = make_observation("temperature_sensors", '{"k10temp":[{"current":54.5,"label":"Tctl"}]}')
    assert validate_fact_grounding(query, [obs])["grounded"]


def test_gmail_scope_requires_inbox_and_unread_filter():
    from tools.grounding import make_observation, validate_fact_grounding
    query = "How many emails in my inbox are unread?"
    raw = '{"ok":true,"messages":[],"query":"in:sent"}'
    assert not validate_fact_grounding(query, [make_observation("gmail_search_messages", raw)])["grounded"]
    raw = '{"ok":true,"messages":[],"query":"in:inbox is:unread"}'
    assert validate_fact_grounding(query, [make_observation("gmail_search_messages", raw)])["grounded"]


def _run_runtime(monkeypatch, replies, query, results=None, protocol="json"):
    from al_agent import turn_engine as te, state
    from tools.conversation_context import conversation_context
    names = ["gmail_inbox_counts", "current_time", "calculate", "get_user_profile"]
    schemas = [{"type": "function", "function": {"name": n, "description": n.replace("_", " "), "parameters": {"type": "object", "properties": {}}}} for n in names]
    monkeypatch.setattr(te, "catalog_snapshot", lambda: (schemas, {n: lambda: None for n in names}, {n: {"readonly": True} for n in names}))
    monkeypatch.setattr(te, "execute_registered_tool", lambda name, args, **kw: (results or {}).get(name, {}))
    monkeypatch.setitem(state.AGENT_CFG, "tool_protocol", protocol)
    monkeypatch.setattr(state, "MODEL_TRACE_ENABLED", False)
    events = []
    client = FakeClient(replies)
    with conversation_context("runtime-audit"), te_context(events):
        te.handle_user_turn([], query, False, refresh_history=True, runtime_overrides={"OLLAMA": client})
    return events, client


def te_context(events):
    from al_agent.events import frontend_event_context
    return frontend_event_context(events.append)


def test_production_turn_rejects_unsupported_final_without_discovery(monkeypatch, isolated_db):
    events, _ = _run_runtime(monkeypatch, ["0 emails"] * 3, "How many emails are in my inbox?", protocol="qwen_xml")
    assert not events[-1]["success"]
    assert any(e["type"] == "assistant_reset" for e in events)
    assert not any("0 emails" in str(e.get("content", "")) for e in events if e["type"] == "assistant_delta")
    history = isolated_db._load_chat_history_from_db(conversation_id="runtime-audit", include_compacted=True)
    assert not any(m["content"] == "0 emails" for m in history)


def test_grounded_qwen_fact_answer_still_streams(monkeypatch, isolated_db):
    replies = ['<tool_call><function=load_tools><parameter=names>["gmail_inbox_counts"]</parameter></function></tool_call>',
               '<tool_call><function=gmail_inbox_counts></function></tool_call>', "You have 27 emails and 3 unread."]
    events, _ = _run_runtime(monkeypatch, replies, "How many emails are in my inbox?",
                             {"gmail_inbox_counts": {"ok": True, "messages_total": 27, "messages_unread": 3}}, protocol="qwen_xml")
    assert events[-1]["success"]
    assert any("27" in e.get("content", "") for e in events if e["type"] == "assistant_delta")


def test_production_turn_recovers_after_missing_evidence(monkeypatch, isolated_db):
    replies = [final("0 emails"), tool("load_tools", names=["gmail_inbox_counts"]), tool("gmail_inbox_counts"), final("You have 27 emails, 3 unread.")]
    events, client = _run_runtime(monkeypatch, replies, "How many emails are in my inbox?", {"gmail_inbox_counts": {"ok": True, "messages_total": 27, "messages_unread": 3}})
    assert events[-1]["success"]
    assert "Missing verified evidence" in json.dumps(client.requests[1])
    from tools.working_state import WorkingStateStore
    assert all(r["satisfied"] for r in WorkingStateStore(conversation_id="runtime-audit").load()["fact_requirements"])


def test_multidomain_turn_requires_each_fact(monkeypatch, isolated_db):
    replies = [tool("load_tools", names=["current_time"]), tool("current_time"), *([final("All done")] * 3)]
    events, _ = _run_runtime(monkeypatch, replies, "Check the current time and count my inbox emails", {"current_time": {"utc": "2026-09-26T00:00:00Z", "local": "2026-09-25T20:00:00-04:00", "timezone": "America/Toronto"}})
    assert not events[-1]["success"]


def test_working_state_cannot_complete_with_pending_facts(isolated_db):
    from tools.working_state import WorkingStateStore
    store = WorkingStateStore(conversation_id="audit")
    store.begin_turn(turn_id=1, objective="Read Gmail", rolling_summary="", recalled_context="", recent_messages=[], policy_note="", tool_schemas=[], fact_requirements=[{"fact_type": "gmail", "status": "pending"}])
    store.complete_turn()
    assert store.load()["status"] != "complete"


def test_unknown_mutation_return_blocks_repeat():
    from al_agent.tool_session import ToolSession
    from test_autonomous_loop import WRITE
    calls = []
    session = ToolSession([WRITE], lambda *args: calls.append(args) or {"ok": False, "outcome_unknown": True}, {"put": {"readonly": False}}, initial_active=["put"])
    session.invoke("put", {"key": "x", "value": 1})
    result = session.invoke("put", {"key": "x", "value": 1})
    assert len(calls) == 1 and result["outcome_unknown"]


def test_schema_order_does_not_depend_on_activation_order():
    from al_agent.tool_session import ToolSession
    from test_autonomous_loop import WRITE, SCHEMA
    session = ToolSession([WRITE, SCHEMA], lambda *args: None, initial_active=["put", "lookup"])
    before = session.schemas
    session.invoke("load_tools", {"names": ["lookup", "put"]})
    assert session.schemas == before


def test_parser_preserves_string_whitespace():
    from al_agent.agent_loop import decode_response
    xml = '<tool_call>\n<function=put>\n<parameter=key>\n  exact  \n</parameter>\n</function>\n</tool_call>'
    assert decode_response(xml, [], "qwen_xml")[1][0]["arguments"]["key"] == "  exact  "


def test_conflicting_native_and_xml_arguments_are_rejected():
    from al_agent.agent_loop import decode_response
    xml = '<tool_call><function=put><parameter=key>safe</parameter></function></tool_call>'
    native = [{"function": {"name": "put", "arguments": {"key": "different"}}}]
    with pytest.raises(ValueError, match="arguments disagree"):
        decode_response(xml, native, "qwen_xml")


@pytest.mark.parametrize("fragment", ["<tool_", "<tool_call", "<tool_call>"])
def test_incomplete_xml_is_not_an_answer(fragment):
    from al_agent.agent_loop import decode_response
    with pytest.raises(ValueError):
        decode_response(fragment, [], "qwen_xml")


def test_cancel_during_prefill_does_not_wait_for_first_chunk():
    from al_agent.model_protocol import consume_chat_stream
    cancel, release = threading.Event(), threading.Event()
    def stream():
        release.wait(2)
        yield {"message": {"content": "late"}, "done": True}
    timer = threading.Timer(0.02, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        capture = consume_chat_stream(stream(), content_stream_allowed=True, leak_detector=lambda _: False, cancel_requested=cancel.is_set, first_chunk_timeout_seconds=30, idle_timeout_seconds=30)
        assert capture.cancelled and time.monotonic() - started < 0.7
    finally:
        release.set()
        timer.join()


def test_soft_budget_preserves_active_request_and_unresolved_state():
    from al_agent.agent_loop import fit_context, LoopConfig
    from tools.context import estimate_prompt_tokens
    cfg = LoopConfig("audit", {"num_ctx": 16384}, reserve_tokens=2560, soft_prompt_tokens=8192, hard_prompt_tokens=13824)
    rows = [{"role": "system", "content": "Unresolved: preserve this requirement"}]
    for i in range(4):
        rows += [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old detail " * 900}]
    rows += [{"role": "user", "content": "exact active request"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}]}, {"role": "tool", "content": "critical active evidence"}]
    fitted = fit_context(rows, [], cfg)
    assert estimate_prompt_tokens(fitted, []) <= 8192
    assert "exact active request" in str(fitted) and "critical active evidence" in str(fitted)
    assert fitted[0]["content"] == rows[0]["content"]


def test_hard_ceiling_refuses_without_truncating_active_request():
    from al_agent.agent_loop import fit_context, LoopConfig, LoopStopped
    rows = [{"role": "system", "content": "policy"}, {"role": "user", "content": "x" * 45000}]
    with pytest.raises(LoopStopped, match="context budget"):
        fit_context(rows, [], LoopConfig("audit", {"num_ctx": 16384}, reserve_tokens=2560, hard_prompt_tokens=13824))
    assert len(rows[-1]["content"]) == 45000


@pytest.mark.parametrize("payload,ok", [({"messagesTotal": 8, "messagesUnread": 2}, True), ({}, False), ({"messagesTotal": 0, "messagesUnread": 0}, True)])
def test_gmail_counts_do_not_guess_zero(monkeypatch, payload, ok):
    from tools import google_workspace as gw
    monkeypatch.setattr(gw, "_get_client", lambda: SimpleNamespace(get=lambda *args, **kwargs: payload))
    from al_agent.agent_loop import tool_outcome
    success, result = tool_outcome(gw.gmail_inbox_counts())
    assert success is ok
    if ok:
        assert result["messages_total"] == payload["messagesTotal"] and not result["count_is_estimate"]


def test_four_recent_turns_survive_hundreds_of_tool_rows(isolated_db):
    from tools.state_tape import compact_recent_conversation
    for i in range(6):
        isolated_db._save_message_to_db({"role": "user", "content": f"Request {i}"}, "audit")
        for _ in range(30):
            isolated_db._save_message_to_db({"role": "assistant", "content": "<tool_call>secret protocol</tool_call>", "tool_calls": [{"function": {"name": "lookup"}}]}, "audit")
            isolated_db._save_message_to_db({"role": "tool", "content": '{"raw_secret":true}'}, "audit")
        isolated_db._save_message_to_db({"role": "assistant", "content": f"Answer {i}"}, "audit")
    surface = compact_recent_conversation(isolated_db.load_recent_conversational_history(4, "audit"))
    assert [m["content"] for m in surface if m["role"] == "user"] == [f"Request {i}" for i in range(2, 6)]
    assert "raw_secret" not in str(surface) and "<tool_call>" not in str(surface)


def test_archived_observations_are_searchable_but_conversation_scoped(isolated_db):
    from tools.conversation_context import conversation_context
    ref = isolated_db.store_tool_observation("temperature_sensors", "CPU sensor Tctl 54.5 C", "audit")
    isolated_db.store_tool_observation("temperature_sensors", "CPU private other conversation", "other")
    isolated_db.store_tool_observation("tool_search", "CPU catalog", "audit")
    with conversation_context("audit"):
        rows = json.loads(isolated_db.search_observations("CPU"))
    assert len(rows) == 1 and rows[0]["observation_id"] == ref


def test_historical_facts_require_original_evidence(isolated_db):
    from tools.grounding import requested_fact_types, make_observation, validate_fact_grounding
    query = "What did we determine about my CPU temperature earlier?"
    assert requested_fact_types(query) == {"historical_evidence"}
    fresh = make_observation("temperature_sensors", '{"cpu":[{"current":40}]}')
    assert not validate_fact_grounding(query, [fresh])["grounded"]
    ref = isolated_db.store_tool_observation("temperature_sensors", '{"cpu":[{"current":54.5}]}')
    original = make_observation("read_observation", isolated_db.read_observation(ref))
    assert validate_fact_grounding(query, [original])["grounded"]
    catalog_ref = isolated_db.store_tool_observation("tool_search", "CPU tools")
    assert not validate_fact_grounding(query, [make_observation("read_observation", isolated_db.read_observation(catalog_ref))])["grounded"]


def test_soft_compaction_keeps_rehydration_handle():
    from al_agent.agent_loop import fit_context, LoopConfig
    rows = [{"role": "system", "content": "policy"}, {"role": "user", "content": "old " * 9000},
            {"role": "assistant", "content": "old answer", "_state_tape_summary": "CPU 54.5 C [observation_id=original]"},
            {"role": "user", "content": "current"}]
    fitted = fit_context(rows, [], LoopConfig("audit", {"num_ctx": 16384}))
    assert "observation_id=original" in str(fitted) and fitted[-1]["content"] == "current"


def test_active_results_not_shortened_after_two_iterations():
    from test_autonomous_loop import run
    critical = "first result " + "x" * 1100 + " downstream critical value=12345"
    replies = [tool("load_tools", names=["lookup"]), tool("lookup", key="a"), tool("lookup", key="b"), tool("lookup", key="c"), final("Done")]
    *_, client, session = run(replies, execute=lambda name, args: critical)
    assert critical in str(client.requests[-1])


def test_json_protocol_is_not_streamed_as_visible_prose():
    from test_autonomous_loop import run
    _, _, _, events, _, _ = run([final("Hello")])
    assert not any(kind == "assistant_delta" for kind, _ in events)
    assert any(kind == "assistant_final" and row["content"] == "Hello" for kind, row in events)


def test_model_trace_redacts_credentials_and_sets_private_permissions(tmp_path):
    from al_agent.model_traces import record_model_trace
    path = tmp_path / "trace.jsonl"
    path.write_text("")
    path.chmod(0o644)
    record_model_trace(path=str(path), enabled=True, max_bytes=1000000, conversation_id="audit", turn_id=1, call_index=1,
                       model="mock", role="main", purpose="test", thinking_enabled=False, tools=[], options={},
                       messages=[{"role": "user", "content": 'Authorization: Bearer secret-bearer\n{"password":"secret-password"}'}],
                       request_extra={"authorization": "secret-header"}, completion={"access_token": "secret-token"})
    assert "secret-" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


def test_full_result_metadata_is_recorded_before_prompt_truncation(monkeypatch, isolated_db):
    # Valid evidence lies beyond the model-facing preview limit.
    result = {"padding": "x" * 7000, "ok": True, "messages_total": 27, "messages_unread": 3}
    replies = [tool("load_tools", names=["gmail_inbox_counts"]), tool("gmail_inbox_counts"), final("27 emails")]
    events, _ = _run_runtime(monkeypatch, replies, "How many emails are in my inbox?", {"gmail_inbox_counts": result})
    assert events[-1]["success"]


def test_disconnect_keeps_running_turn_cancellable(monkeypatch):
    import asyncio
    from fastapi import WebSocketDisconnect
    from webui import chat, server
    from al_agent.events import emit_event
    disconnected = threading.Event()
    release = threading.Event()

    class Socket:
        async def send_json(self, packet):
            if packet["type"] != "accepted":
                disconnected.set()
                raise WebSocketDisconnect()

    def work(*args, **kwargs):
        emit_event("assistant_delta", content="Started")
        assert release.wait(3)

    monkeypatch.setattr(chat, "ensure_conversation", lambda cid: None)
    monkeypatch.setattr(chat, "_workspace_file_snapshot", lambda: {})
    monkeypatch.setattr(chat, "_new_artifacts", lambda *a, **kw: [])
    monkeypatch.setattr(chat.agent_runtime, "handle_user_turn", work)

    async def run():
        task = asyncio.create_task(chat._run_turn(Socket(), {"turn_id": "disconnect-audit", "content": "Hello"}))
        try:
            assert await asyncio.to_thread(disconnected.wait, 2)
            assert server.turn_status("disconnect-audit")["active"]
            assert server.cancel("disconnect-audit")["ok"]
        finally:
            release.set()
            await task
        assert not server.turn_status("disconnect-audit")["active"]
    asyncio.run(run())


def test_history_cache_expires_after_external_writer(isolated_db, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(isolated_db.time, "monotonic", lambda: clock[0])
    isolated_db._save_message_to_db({"role": "user", "content": "original"}, "audit")
    assert len(isolated_db._load_chat_history_from_db(conversation_id="audit")) == 1
    with isolated_db._connect() as conn:
        conn.execute("INSERT INTO chat_history(role,content,conversation_id) VALUES ('assistant','external write','audit')")
    clock[0] += 1
    assert len(isolated_db._load_chat_history_from_db(conversation_id="audit")) == 2


def test_historical_page_provenance_matches_requested_domain(isolated_db):
    from tools.grounding import make_observation, validate_fact_grounding
    query = "What was the CPU temperature earlier?"
    mailbox = isolated_db.store_tool_observation("gmail_inbox_counts", '{"ok":true,"messages_total":27}')
    assert not validate_fact_grounding(query, [make_observation("read_observation", isolated_db.read_observation(mailbox))])["grounded"]
    raw = '{"padding":"' + "x" * 5500 + '","cpu":[{"current":54.5}]}'
    ref = isolated_db.store_tool_observation("temperature_sensors", raw)
    page = isolated_db.read_observation(ref, offset=5500, length=100)
    assert validate_fact_grounding(query, [make_observation("read_observation", page)])["grounded"]


def test_process_io_reports_unavailable_without_fake_counters(monkeypatch):
    from tools.primitive_modules import process
    from al_agent.agent_loop import tool_outcome
    monkeypatch.setattr(process.psutil, "Process", lambda pid: SimpleNamespace())
    ok, result = tool_outcome(process.process_io(1))
    assert not ok and result["supported"] is False and "read_bytes" not in result


def test_existing_simulator_actually_dispatches_tools(capsys):
    from diagnostics import simulate_turns
    simulate_turns.main()
    report = json.loads(capsys.readouterr().out)
    assert [row["model_calls"] for row in report["results"]] == [1, 3, 4]
    assert report["results"][1]["executed_tools"] == ["calculate"]


def test_browser_reconnect_recovers_history_without_resubmitting():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser recovery unit test")
    source = (Path(__file__).resolve().parents[1] / "webui/static/app.js").read_text()
    implementation = source[source.index("async function recoverActiveTurn("):source.index("function connect(){")]
    script = """
const assert = require('node:assert/strict');
let activeTurn='turn', socket={readyState:1}, WebSocket={OPEN:1};
let timers=[], active=true, history=0, conversations=0, finished=0, requests=[];
const api=async(path)=>{requests.push(path);return {active};};
const setStatus=()=>{};
const setTimeout=(fn)=>timers.push(fn);
const finishTurn=()=>{finished++;activeTurn=null;};
const loadHistory=async()=>{history++;};
const loadConversations=async()=>{conversations++;};
""" + implementation + """
(async()=>{
  await recoverActiveTurn('turn');
  assert.equal(finished,0); assert.equal(timers.length,1);
  active=false; await timers.shift()();
  assert.equal(finished,1); assert.equal(history,1); assert.equal(conversations,1);
  assert.deepEqual(requests,['/api/turns/turn','/api/turns/turn']);
  await recoverActiveTurn('turn'); assert.equal(requests.length,2);
})().catch(error=>{console.error(error);process.exit(1);});
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
