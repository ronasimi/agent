from tools.context import build_active_messages, estimate_tokens


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
