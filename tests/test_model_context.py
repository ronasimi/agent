from tools.model_context import SharedModelContext


def test_shared_context_contains_semantic_state_but_not_tool_observations():
    bridge = SharedModelContext(
        request="continue the earlier task",
        summary="We are editing the agent harness.",
        relevant_memory="Use the local models.",
        recent_messages=[
            {"role": "user", "content": "inspect config"},
            {"role": "tool", "content": "SECRET TOOL OUTPUT SHOULD NOT BE BRIDGED"},
            {"role": "assistant", "content": "I will inspect it."},
        ],
        tool_schemas=[{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a workspace file.",
                "parameters": {"type": "object", "required": ["filename"]},
            },
        }],
        max_chars=5000,
    )
    text = bridge.render()
    assert "editing the agent harness" in text
    assert "inspect config" in text
    assert "read_file" in text
    assert "SECRET TOOL OUTPUT" not in text


def test_shared_context_round_trips_structured_validator_state():
    bridge = SharedModelContext(request="do work")
    bridge.add_validator_event(
        {"decision": "switch_tool", "diagnosis": "wrong_tool", "suggested_tool": "read_file"},
        {"kind": "tool_failure", "key": "execute_shell", "attempts": 3},
    )
    text = bridge.render()
    assert "wrong_tool" in text
    assert "read_file" in text
    assert "execute_shell" in text


def test_shared_context_is_bounded():
    bridge = SharedModelContext(
        request="x" * 3000,
        summary="s" * 5000,
        relevant_memory="m" * 5000,
        recent_messages=[{"role": "user", "content": "r" * 5000}],
        max_chars=2400,
    )
    assert len(bridge.render()) <= 2400
