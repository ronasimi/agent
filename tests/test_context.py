from tools.context import (
    build_active_messages,
    compaction_cutoff_id,
    estimate_tokens,
    fit_tool_loop_messages,
    split_turns,
)


def test_context_is_bounded():
    messages = [{"role": "user", "content": "x " * 1000}, {"role": "assistant", "content": "y " * 1000}]
    result = build_active_messages(
        system_prompt="system",
        summary="summary",
        history=messages,
        max_ctx_tokens=1000,
        reserve_tokens=200,
        recent_messages=2,
    )
    assert result[0]["role"] == "system"
    assert len(result) <= 4
    assert estimate_tokens(result[-1]["content"]) < 1000


def test_extra_prompt_budget_is_reserved():
    result = build_active_messages(
        system_prompt="system",
        summary="",
        history=[{"role": "user", "content": "x " * 1000}],
        max_ctx_tokens=1000,
        reserve_tokens=200,
        recent_messages=2,
        extra_prompt_tokens=300,
    )
    assert estimate_tokens(result[0]["content"]) < 1000
    assert sum(estimate_tokens(m.get("content", "")) for m in result) <= 800


def test_turn_grouping_preserves_tool_transaction():
    history = [
        {"role": "user", "content": "first", "_db_id": 1},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}], "_db_id": 2},
        {"role": "tool", "content": "result", "tool_call_id": "c1", "_db_id": 3},
        {"role": "assistant", "content": "done", "_db_id": 4},
        {"role": "user", "content": "second", "_db_id": 5},
    ]
    turns = split_turns(history)
    assert [len(turn) for turn in turns] == [4, 1]
    assert "_db_id" not in turns[0][0]


def test_compaction_cutoff_keeps_whole_recent_turn():
    history = [
        {"role": "user", "content": "one", "_db_id": 1},
        {"role": "assistant", "content": "answer", "_db_id": 2},
        {"role": "user", "content": "two", "_db_id": 3},
        {"role": "assistant", "content": "call", "_db_id": 4},
        {"role": "tool", "content": "result", "_db_id": 5},
        {"role": "assistant", "content": "answer", "_db_id": 6},
    ]
    assert compaction_cutoff_id(history, keep_messages=3) == 2


def test_tool_loop_tail_is_bounded_without_dropping_prefix():
    prefix = [{"role": "system", "content": "system"}, {"role": "user", "content": "request"}]
    tail = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "one"}]},
        {"role": "tool", "content": "x " * 2000, "tool_call_id": "one"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "two"}]},
        {"role": "tool", "content": "y " * 2000, "tool_call_id": "two"},
    ]
    result = fit_tool_loop_messages(prefix, tail, max_ctx_tokens=1000, reserve_tokens=100)
    assert result[:2] == prefix
    assert sum(estimate_tokens(message.get("content", "")) for message in result) <= 900


def test_image_inputs_reserve_context_budget():
    from tools.context import IMAGE_TOKEN_ESTIMATE, estimate_messages_tokens

    without_image = estimate_messages_tokens([{"role": "user", "content": "describe this"}])
    with_image = estimate_messages_tokens([{"role": "user", "content": "describe this", "images": ["abc"]}])
    assert with_image - without_image == IMAGE_TOKEN_ESTIMATE


def test_working_tool_tail_drops_old_raw_results_but_keeps_latest_transactions():
    from tools.context import compact_working_tool_tail

    tail = []
    for i in range(4):
        cid = f"c{i}"
        tail.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "function": {"name": "demo", "arguments": {"i": i}}}]})
        tail.append({"role": "tool", "content": f"result-{i}", "tool_call_id": cid})
    compact = compact_working_tool_tail(tail, keep_tool_results=2)
    text = " ".join(str(item.get("content", "")) for item in compact)
    assert "result-0" not in text and "result-1" not in text
    assert "result-2" in text and "result-3" in text
    kept_ids = {item.get("tool_call_id") for item in compact if item.get("role") == "tool"}
    assert kept_ids == {"c2", "c3"}


def test_one_turn_working_state_context_excludes_previous_task_raw_history():
    history = [
        {"role": "user", "content": "old validator task"},
        {"role": "assistant", "content": "validator_test.txt"},
        {"role": "tool", "content": "VALIDATOR_RECOVERY_74291", "tool_call_id": "old"},
        {"role": "assistant", "content": "old done"},
        {"role": "user", "content": "new health assessment"},
    ]
    active = build_active_messages(
        system_prompt="system",
        summary="",
        history=history,
        max_ctx_tokens=16000,
        reserve_tokens=2000,
        working_state='{"objective":"new health assessment"}',
        max_history_turns=1,
    )
    rendered = "\n".join(str(item.get("content", "")) for item in active)
    assert "new health assessment" in rendered
    assert "validator_test.txt" not in rendered
    assert "VALIDATOR_RECOVERY_74291" not in rendered


def test_working_tool_tail_pins_latest_media_across_later_tools():
    from tools.context import compact_working_tool_tail

    tail = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "shot", "function": {"name": "take_web_screenshot", "arguments": {}}}]},
        {"role": "tool", "content": "saved screenshot", "tool_call_id": "shot"},
        {"role": "user", "content": "inspect actual media", "images": ["base64-image"]},
    ]
    for i in range(4):
        cid = f"later-{i}"
        tail.append({"role": "assistant", "content": "", "tool_calls": [{"id": cid, "function": {"name": "demo", "arguments": {}}}]})
        tail.append({"role": "tool", "content": f"later result {i}", "tool_call_id": cid})
    compact = compact_working_tool_tail(tail, keep_tool_results=2)
    media = [item for item in compact if item.get("images")]
    assert len(media) == 1
    assert media[0]["images"] == ["base64-image"]
