import json
import tempfile
from pathlib import Path


def _store(monkeypatch, directory: str, **limits):
    from tools import working_state
    db = str(Path(directory) / "state.db")
    monkeypatch.setattr(working_state, "DB_PATH", db)
    return working_state.WorkingStateStore(limits=limits)


def _schema(name="read_file"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read a workspace file.",
            "parameters": {"type": "object", "required": ["filename"]},
        },
    }


def test_working_state_persists_and_resets_per_turn(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td)
        store.begin_turn(
            turn_id=7,
            objective="Find validator_test.txt. Do not invent contents.",
            rolling_summary="Prior setup.",
            recalled_context="Relevant memory.",
            recent_messages=[{"role": "assistant", "content": "Earlier context"}],
            policy_note="execute_shell is delayed",
            tool_schemas=[_schema()],
        )
        store.record_tool_result(
            tool_name="read_file",
            arguments={"filename": "/wrong"},
            status="error",
            reason="tool_reported_error",
            result_text="File was not found",
            fingerprint="abc",
        )
        reloaded = _store(monkeypatch, td).load()
        assert reloaded["turn_id"] == 7
        assert reloaded["failed_approaches"][0]["tool"] == "read_file"
        assert any("Do not invent contents" in item for item in reloaded["constraints"])

        store.begin_turn(
            turn_id=8,
            objective="New task",
            rolling_summary="Prior setup.",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[_schema()],
        )
        fresh = store.load()
        assert fresh["turn_id"] == 8
        assert fresh["failed_approaches"] == []
        assert fresh["objective"] == "New task"


def test_working_state_records_evidence_and_structured_validator(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td, max_render_chars=7000)
        store.begin_turn(
            turn_id=1,
            objective="Inspect a file",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[_schema()],
        )
        store.record_tool_result(
            tool_name="read_file",
            arguments={"filename": "x"},
            status="ok",
            reason="ok",
            result_text="actual filesystem evidence",
            fingerprint="deadbeef",
        )
        store.record_validator(
            {"decision": "switch_tool", "diagnosis": "wrong_tool", "suggested_tool": "repo_status"},
            {"kind": "tool_failure", "key": "read_file", "attempts": 3},
        )
        persisted = store.load()
        assert "actual filesystem evidence" in persisted["verified_observations"][0]["evidence_preview"]
        text = store.render()
        parsed = json.loads(text)
        assert parsed["verified_observations"][0]["tool"] == "read_file"
        assert "evidence_preview" not in parsed["verified_observations"][0]
        assert "actual filesystem evidence" not in text
        assert parsed["validator_history"][0]["diagnosis"] == "wrong_tool"
        assert parsed["current_plan"][0]["tool"] == "repo_status"


def test_working_state_is_bounded(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td, max_render_chars=2200, background_chars=1000)
        store.begin_turn(
            turn_id=1,
            objective="x" * 5000,
            rolling_summary="s" * 5000,
            recalled_context="m" * 5000,
            recent_messages=[{"role": "user", "content": "r" * 5000}],
            policy_note="",
            tool_schemas=[_schema()],
        )
        rendered = store.render()
        assert len(rendered) <= 2200
        assert json.loads(rendered)["turn_id"] == 1


def test_active_context_uses_working_state_instead_of_old_history():
    from tools.context import build_active_messages

    history = [
        {"role": "user", "content": "OLD UNIQUE CONTEXT"},
        {"role": "assistant", "content": "OLD ANSWER"},
        {"role": "user", "content": "CURRENT REQUEST"},
    ]
    result = build_active_messages(
        system_prompt="stable-system",
        summary="SHOULD NOT APPEAR",
        history=history,
        max_ctx_tokens=4000,
        reserve_tokens=500,
        working_state='{"objective":"CURRENT REQUEST","background":{"recent_context":"OLD UNIQUE CONTEXT"}}',
        max_history_turns=1,
    )
    joined = "\n".join(str(m.get("content", "")) for m in result)
    assert "SHOULD NOT APPEAR" not in joined
    assert joined.count("OLD UNIQUE CONTEXT") == 1
    assert "CURRENT REQUEST" in result[-1]["content"]

def test_clear_chat_history_clears_working_state(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        from tools import memory, runtime, working_state
        db = str(Path(td) / "shared.db")
        monkeypatch.setattr(runtime, "DB_PATH", db)
        monkeypatch.setattr(memory, "DB_PATH", db)
        monkeypatch.setattr(working_state, "DB_PATH", db)
        memory.init_db()
        store = working_state.WorkingStateStore()
        store.begin_turn(
            turn_id=9,
            objective="temporary task",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[_schema()],
        )
        assert store.load()["turn_id"] == 9
        memory.clear_chat_history()
        assert store.load()["turn_id"] == 0


def test_new_task_epoch_drops_stale_recent_context_and_followup_carries_evidence(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td)
        store.begin_turn(
            turn_id=1,
            objective="Old validator task",
            rolling_summary="old summary",
            recalled_context="",
            recent_messages=[{"role": "assistant", "content": "validator_test.txt"}],
            policy_note="",
            tool_schemas=[_schema()],
        )
        store.record_tool_result(
            tool_name="read_file", arguments={"filename": "validator_test.txt"}, status="ok", reason="ok",
            result_text="VALIDATOR_RECOVERY_74291", fingerprint="old",
        )
        epoch1 = store.load()["task_epoch"]

        store.begin_turn(
            turn_id=2,
            objective="Perform a new host health assessment",
            rolling_summary="old summary mentioning validator_test.txt",
            recalled_context="",
            recent_messages=[{"role": "assistant", "content": "validator_test.txt"}],
            policy_note="",
            tool_schemas=[_schema()],
            continuation=False,
        )
        fresh = store.load()
        assert fresh["task_epoch"] == epoch1 + 1
        assert fresh["verified_observations"] == []
        assert fresh["background"]["recent_context"] == ""
        assert fresh["background"]["rolling_summary"] == ""

        store.record_tool_result(
            tool_name="read_file", arguments={"filename": "host.txt"}, status="ok", reason="ok",
            result_text="host evidence", fingerprint="host",
        )
        store.begin_turn(
            turn_id=3,
            objective="Without rerunning tools, summarize those results",
            rolling_summary="host summary",
            recalled_context="",
            recent_messages=[{"role": "assistant", "content": "host health result"}],
            policy_note="",
            tool_schemas=[_schema()],
            continuation=True,
        )
        follow = store.load()
        assert follow["task_epoch"] == fresh["task_epoch"]
        assert follow["verified_observations"][0]["tool"] == "read_file"
        assert "host health result" in follow["background"]["recent_context"]


def test_evidence_digest_is_valid_json_and_contains_untrusted_preview(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td, evidence_render_chars=500)
        store.begin_turn(
            turn_id=1, objective="inspect", rolling_summary="", recalled_context="", recent_messages=[],
            policy_note="", tool_schemas=[_schema()],
        )
        for i in range(8):
            store.record_tool_result(
                tool_name="read_file", arguments={"filename": str(i)}, status="ok", reason="ok",
                result_text=(f"evidence-{i} " * 80), fingerprint=str(i),
            )
        digest = store.render_evidence(500)
        parsed = json.loads(digest)
        assert isinstance(parsed, list)
        assert len(digest) <= 500


def test_followup_preserves_prior_requirement_status_for_state_visibility(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td)
        reqs = [{"key": "dns", "tool": "dns_diagnose", "label": "DNS", "status": "satisfied", "attempts": 1}]
        store.begin_turn(
            turn_id=1, objective="check dns", rolling_summary="", recalled_context="", recent_messages=[],
            policy_note="", tool_schemas=[_schema()], requirements=reqs, continuation=False,
        )
        store.begin_turn(
            turn_id=2, objective="Without rerunning, summarize what remains", rolling_summary="summary", recalled_context="",
            recent_messages=[], policy_note="", tool_schemas=[_schema()], requirements=[], continuation=True,
        )
        state = store.load()
        assert state["requirements"][0]["tool"] == "dns_diagnose"
        assert state["requirements"][0]["status"] == "satisfied"


def test_requirement_persistence_limit_is_separate_from_prompt_render_limit(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td, max_render_chars=50000)
        requirements = [
            {
                "key": f"tooltest:{i:02d}",
                "tool": f"tool_{i}",
                "label": f"requirement {i}",
                "status": "satisfied",
                "attempts": 1,
                "last_reason": "ok",
                "fingerprint": f"fp{i}",
                "scope": {"item_number": i},
            }
            for i in range(1, 38)
        ]
        store.begin_turn(
            turn_id=37,
            objective="37-item deterministic plan",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[_schema()],
            requirements=requirements,
        )
        persisted = store.load()
        assert len(persisted["requirements"]) == 37
        assert persisted["requirements"][0]["key"] == "tooltest:01"
        assert persisted["requirements"][-1]["key"] == "tooltest:37"

        # Prompt rendering remains separately bounded, so preserving the complete
        # audit ledger on disk does not automatically inflate model context.
        rendered = json.loads(store.render())
        assert len(rendered["requirements"]) <= store.limits["requirement_items"]


def test_requirement_evidence_provenance_persists(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td, max_render_chars=50000)
        requirements = [{
            "key": "tooltest:12",
            "tool": "__tooltest_observation_capability__",
            "label": "observation retrieval capability discovery",
            "status": "satisfied",
            "attempts": 1,
            "last_reason": "tool_search discovered read_observation",
            "fingerprint": "",
            "scope": {"derived": True},
            "evidence": [{
                "source": "tool_call",
                "tool": "tool_search",
                "status": "ok",
                "reason": "ok",
                "arguments_digest": "abc123",
                "evidence_ref": "obs-1",
            }],
        }]
        store.begin_turn(
            turn_id=12,
            objective="provenance",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[_schema()],
            requirements=requirements,
        )
        row = store.load()["requirements"][0]
        assert row["attempts"] == 1
        assert row["evidence"][0]["tool"] == "tool_search"
        assert row["evidence"][0]["evidence_ref"] == "obs-1"


def test_requirement_persistence_keeps_72_item_generalized_recipe_plan(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td, max_render_chars=50000)
        requirements = [
            {
                "key": f"genrecipe:{i:02d}",
                "tool": f"tool_{i}",
                "label": f"requirement {i}",
                "status": "satisfied",
                "attempts": 1,
                "last_reason": "ok",
                "fingerprint": f"fp{i}",
                "scope": {"item_number": i},
                "evidence": [{"source": "test", "status": "ok"}],
            }
            for i in range(1, 73)
        ]
        store.begin_turn(
            turn_id=72,
            objective="72-item generalized recipe plan",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[_schema()],
            requirements=requirements,
        )
        persisted = store.load()
        assert len(persisted["requirements"]) == 72
        assert persisted["requirements"][24]["key"] == "genrecipe:25"
        assert persisted["requirements"][-1]["key"] == "genrecipe:72"
        rendered = json.loads(store.render())
        assert len(rendered["requirements"]) <= 24


def test_requirement_evidence_preview_persists_but_is_omitted_from_model_render(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        store = _store(monkeypatch, td, max_render_chars=7000)
        requirement = {
            "key": "current_time",
            "tool": "current_time",
            "label": "current time",
            "status": "satisfied",
            "attempts": 1,
            "last_reason": "ok",
            "fingerprint": "clock",
            "scope": {},
            "evidence": [{
                "source": "tool_call",
                "tool": "current_time",
                "status": "ok",
                "reason": "ok",
                "arguments_digest": "abc123",
                "evidence_ref": "obs-clock",
                "evidence_preview": "durable clock evidence 2026-09-23T16:22:36-04:00",
            }],
        }
        store.begin_turn(
            turn_id=1,
            objective="What time is it?",
            rolling_summary="",
            recalled_context="",
            recent_messages=[],
            policy_note="",
            tool_schemas=[],
            requirements=[requirement],
        )
        persisted = store.load()["requirements"][0]
        assert persisted["evidence"][0]["evidence_ref"] == "obs-clock"
        assert "durable clock evidence" in persisted["evidence"][0]["evidence_preview"]
        rendered = json.loads(store.render())
        assert rendered["requirements"][0]["evidence"][0]["evidence_ref"] == "obs-clock"
        assert "evidence_preview" not in rendered["requirements"][0]["evidence"][0]
