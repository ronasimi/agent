from __future__ import annotations


def test_system_policy_uses_state_tape_before_declaring_history_unavailable():
    from al_agent.prompts import SYSTEM_POLICY

    assert "Harness State Tape and rolling summary" in SYSTEM_POLICY
    assert "Before saying historical information is unavailable" in SYSTEM_POLICY
    assert "read_observation" in SYSTEM_POLICY
    assert "search_conversation_history" in SYSTEM_POLICY
    assert "search_memory" in SYSTEM_POLICY


def test_system_policy_routes_arbitrary_iteration_to_durable_compute():
    from al_agent.prompts import SYSTEM_POLICY

    assert "arbitrary number of state transitions" in SYSTEM_POLICY
    assert "start_computation" in SYSTEM_POLICY
    assert "get_computation_status" in SYSTEM_POLICY
    assert "until HALT or cancellation" in SYSTEM_POLICY


def test_state_tape_retains_observation_handle_without_raw_payload(tmp_path, monkeypatch):
    from tools import memory, runtime
    from tools.state_tape import CompactToolOutcome, StateTapeStore

    db = str(tmp_path / "state-tape.db")
    monkeypatch.setattr(runtime, "DB_PATH", db)
    monkeypatch.setattr(memory, "DB_PATH", db)
    memory.init_db()
    conversation = memory.create_conversation("Tape evidence")
    tape = StateTapeStore(conversation["id"])
    observation_id = "a" * 32
    tape.commit_turn(
        turn_id=7,
        source_message_id=0,
        objective="inspect host memory",
        assistant_text="Memory checked.",
        outcomes=[
            CompactToolOutcome(
                tool="memory_info",
                ok=True,
                summary="Memory query succeeded: 62% used.",
                observation_id=observation_id,
            )
        ],
        status="complete",
    )
    rendered = tape.render_prompt_context()
    assert f"observation_id={observation_id}" in rendered
    assert '"memory"' not in rendered
    assert "tool_response" not in rendered


def test_practical_turing_completeness_contract():
    from diagnostics.check_turing_completeness import check_contract

    assert check_contract() == []
