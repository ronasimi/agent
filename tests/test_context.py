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
