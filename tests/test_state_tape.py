from __future__ import annotations

import json
import uuid


def _temp_db(monkeypatch, tmp_path):
    from tools import memory, runtime

    db = str(tmp_path / "state-tape.db")
    monkeypatch.setattr(runtime, "DB_PATH", db)
    monkeypatch.setattr(memory, "DB_PATH", db)
    memory._INITIALIZED_DB_PATHS.clear()
    memory._invalidate_history_cache()
    memory._invalidate_context_cache()
    memory.init_db()
    return memory


def test_recent_conversation_strips_historical_tool_protocol():
    from tools.state_tape import compact_recent_conversation

    history = [
        {"role": "user", "content": "Check weather"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "weather_forecast", "arguments": {}}}],
        },
        {
            "role": "tool",
            "tool_name": "weather_forecast",
            "tool_call_id": "c1",
            "content": json.dumps({"current": {"temperature_2m": 17}, "daily": ["x" * 4000]}),
        },
        {"role": "assistant", "content": "It is 17°C."},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "You're welcome."},
    ]
    compact = compact_recent_conversation(history, max_turns=3)
    assert compact == [
        {"role": "user", "content": "Check weather"},
        {"role": "assistant", "content": "It is 17°C."},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "You're welcome."},
    ]
    assert "tool_calls" not in json.dumps(compact)
    assert "daily" not in json.dumps(compact)


def test_large_weather_result_becomes_compact_state_tape(monkeypatch, tmp_path):
    memory = _temp_db(monkeypatch, tmp_path)
    from tools.state_tape import StateTapeStore

    cid = "tape-" + uuid.uuid4().hex
    memory.ensure_conversation(cid)
    user_id = memory._save_message_to_db({"role": "user", "content": "Weather in London"}, cid)
    final_id = memory._save_message_to_db({"role": "assistant", "content": "17°C and cloudy."}, cid)
    raw = json.dumps({
        "provider": "Open-Meteo",
        "latitude": 42.98,
        "longitude": -81.25,
        "current": {"time": "2026-09-25T18:45", "temperature_2m": 17.0, "weather_code": 3},
        "daily": {"blob": "x" * 6000},
    })
    outcome = StateTapeStore.collapse_tool_result(
        tool_name="weather_forecast",
        arguments={"latitude": 42.98, "longitude": -81.25},
        result_text=raw,
        status="ok",
        observation_id="obs1",
    )
    tape = StateTapeStore(cid, recent_entries=6)
    tape.commit_turn(
        turn_id=user_id,
        source_message_id=final_id,
        objective="Weather in London",
        assistant_text="17°C and cloudy.",
        outcomes=[outcome],
        status="complete",
    )
    context = tape.render_prompt_context()
    assert "17.0°C" in context and "cloudy" in context
    assert "Open-Meteo" in context
    assert '"blob"' not in context and ("x" * 100) not in context
    assert len(context) < 1200


def test_unresolved_state_survives_until_matching_turn_completes(monkeypatch, tmp_path):
    memory = _temp_db(monkeypatch, tmp_path)
    from tools.state_tape import StateTapeStore

    cid = "unresolved-" + uuid.uuid4().hex
    tape = StateTapeStore(cid)
    first = memory._save_message_to_db({"role": "user", "content": "Grab a screenshot of cnn.com"}, cid)
    tape.commit_turn(
        turn_id=first,
        source_message_id=first,
        objective="Grab a screenshot of cnn.com",
        assistant_text="",
        outcomes=[],
        status="blocked",
        failure_text="Timed out waiting for the model",
    )
    assert "UNRESOLVED" in tape.render_prompt_context()

    second = memory._save_message_to_db({"role": "user", "content": "Grab a screenshot of cnn.com again"}, cid)
    tape.commit_turn(
        turn_id=second,
        source_message_id=second,
        objective="Grab a screenshot of cnn.com again",
        assistant_text="Screenshot captured.",
        outcomes=[],
        status="complete",
    )
    assert tape.unresolved() == []
    assert "UNRESOLVED" not in tape.render_prompt_context()


def test_old_tape_entries_roll_into_deep_summary_and_advance_watermark(monkeypatch, tmp_path):
    memory = _temp_db(monkeypatch, tmp_path)
    from tools.state_tape import StateTapeStore

    cid = "rollup-" + uuid.uuid4().hex
    tape = StateTapeStore(cid, recent_entries=2, rolling_summary_chars=1200)
    message_ids = []
    for index in range(4):
        user_id = memory._save_message_to_db({"role": "user", "content": f"request {index}"}, cid)
        final_id = memory._save_message_to_db({"role": "assistant", "content": f"answer {index}"}, cid)
        message_ids.append(final_id)
        tape.commit_turn(
            turn_id=user_id,
            source_message_id=final_id,
            objective=f"request {index}",
            assistant_text=f"answer {index}",
            outcomes=[],
            status="complete",
        )
    assert len(tape.recent()) == 2
    assert "request 0" in memory.get_conversation_summary(cid)
    assert memory.get_compacted_through_id(cid) >= message_ids[1]
    visible = memory._load_chat_history_from_db(limit=100, conversation_id=cid)
    assert all(row["_db_id"] > memory.get_compacted_through_id(cid) for row in visible)


def test_active_turn_protocol_collapses_to_prompt_safe_surface(monkeypatch, tmp_path):
    memory = _temp_db(monkeypatch, tmp_path)
    from tools.state_tape import StateTapeStore, compact_recent_conversation

    cid = "surface-" + uuid.uuid4().hex
    user_id = memory._save_message_to_db({"role": "user", "content": "Calculate 6 times 7"}, cid)
    memory._save_message_to_db({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "calculate", "arguments": {"expression": "6*7"}}}],
    }, cid)
    memory._save_message_to_db({
        "role": "tool", "tool_name": "calculate", "tool_call_id": "c1",
        "content": json.dumps({"expression": "6*7", "result": 42}),
    }, cid)
    final_id = memory._save_message_to_db({"role": "assistant", "content": "42"}, cid)

    outcome = StateTapeStore.collapse_tool_result(
        tool_name="calculate", arguments={"expression": "6*7"},
        result_text=json.dumps({"expression": "6*7", "result": 42}), status="ok",
    )
    tape = StateTapeStore(cid)
    tape.commit_turn(
        turn_id=user_id, source_message_id=final_id, objective="Calculate 6 times 7",
        assistant_text="42", outcomes=[outcome], status="complete",
    )
    history = memory._load_chat_history_from_db(limit=100, include_compacted=True, conversation_id=cid)
    surface = compact_recent_conversation(history, max_turns=3)
    rendered = json.dumps(surface, ensure_ascii=False) + tape.render_prompt_context()
    assert all(row.get("role") != "tool" for row in surface)
    assert all(not row.get("tool_calls") for row in surface)
    assert "42" in rendered
    assert "additionalProperties" not in rendered
    assert "tool_call_id" not in rendered


def test_unverified_assistant_claim_is_labeled_non_evidence_in_tape(monkeypatch, tmp_path):
    memory = _temp_db(monkeypatch, tmp_path)
    from tools.state_tape import StateTapeStore

    cid = "unverified-" + uuid.uuid4().hex
    tape = StateTapeStore(cid)
    user_id = memory._save_message_to_db(
        {"role": "user", "content": "Can you access Gmail?"}, cid
    )
    final_id = memory._save_message_to_db(
        {"role": "assistant", "content": "No Gmail account is configured."}, cid
    )
    tape.commit_turn(
        turn_id=user_id,
        source_message_id=final_id,
        objective="Can you access Gmail?",
        assistant_text="No Gmail account is configured.",
        outcomes=[],
        status="complete",
    )
    context = tape.render_prompt_context()
    assert "No Gmail account is configured" in context
    assert "Unverified conversational record (not tool evidence)" in context


def test_recent_conversation_keeps_four_turns_and_removes_formatting_bloat():
    from tools.state_tape import compact_recent_conversation

    history = []
    for i in range(1, 6):
        history.extend([
            {"role": "user", "content": f"Question {i}?"},
            {
                "role": "assistant",
                "content": (
                    f"## Answer {i}\n\n"
                    "| Step | Tool | Result | Status |\n"
                    "|---|---|---|---|\n"
                    f"| {i} | `demo_tool` | **value {i}** | ✅ Success |\n"
                    "```text\nformatted payload\n```"
                ),
            },
        ])
    compact = compact_recent_conversation(
        history, max_turns=4, max_user_chars=500, max_assistant_chars=500
    )
    users = [row["content"] for row in compact if row["role"] == "user"]
    assistants = [row["content"] for row in compact if row["role"] == "assistant"]
    assert users == ["Question 2?", "Question 3?", "Question 4?", "Question 5?"]
    assert len(assistants) == 4
    rendered = "\n".join(assistants)
    assert "##" not in rendered
    assert "```" not in rendered
    assert "|---|" not in rendered
    assert "demo_tool" in rendered
    assert all(len(row) <= 500 for row in assistants)


def test_state_tape_can_skip_turns_already_kept_as_conversation(monkeypatch, tmp_path):
    memory = _temp_db(monkeypatch, tmp_path)
    from tools.state_tape import StateTapeStore

    cid = "dedupe-" + uuid.uuid4().hex
    memory.ensure_conversation(cid)
    tape = StateTapeStore(cid, recent_entries=6, rolling_summary_chars=2400)
    for index in range(6):
        user_id = memory._save_message_to_db(
            {"role": "user", "content": f"question {index}"}, cid
        )
        final_id = memory._save_message_to_db(
            {"role": "assistant", "content": f"answer {index}"}, cid
        )
        tape.commit_turn(
            turn_id=user_id, source_message_id=final_id, objective=f"question {index}",
            assistant_text=f"answer {index}", outcomes=[], status="complete",
        )

    recent = tape.recent()
    overlap = {entry.source_message_id for entry in recent[-4:]}
    rendered = tape.render_prompt_context(exclude_source_message_ids=overlap)
    # Four newest turns live in natural conversation and are not replayed again.
    assert "question 5" not in rendered
    assert "question 4" not in rendered
    assert "question 3" not in rendered
    assert "question 2" not in rendered
    assert "question 0" in rendered or "question 1" in rendered


def test_default_context_config_targets_16k_with_prefill_soft_budget():
    from tools.config import normalize_config

    cfg = normalize_config({"agent": {"model": "agent-main:4b"}})["agent"]
    assert cfg["main_options"]["num_ctx"] == 16384
    assert cfg["context"]["soft_prompt_tokens"] == 8192
    assert cfg["context"]["hard_prompt_tokens"] == 13824
    assert cfg["context"]["recent_conversation_turns"] == 4
