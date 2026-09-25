from al_agent.model_protocol import (
    canonicalize_system_messages,
    consume_chat_stream,
    is_prompt_protocol_error,
    is_retryable_transport_error,
    merge_stream_tool_calls,
    ollama_wire_messages,
    stream_with_preflight_retry,
    tool_result_message,
)
from tools.context import model_message


def _call(name: str, value: int, call_id: str = ""):
    row = {"function": {"name": name, "arguments": {"value": value}}}
    if call_id:
        row["id"] = call_id
    return row


def test_streamed_tool_calls_are_accumulated_across_chunks():
    calls = merge_stream_tool_calls([], [_call("first", 1, "a")])
    calls = merge_stream_tool_calls(calls, [_call("second", 2, "b")])
    assert [item["function"]["name"] for item in calls] == ["first", "second"]


def test_repeated_streamed_tool_call_is_not_duplicated():
    first = _call("demo", 1, "same")
    calls = merge_stream_tool_calls([], [first])
    calls = merge_stream_tool_calls(calls, [first])
    assert len(calls) == 1


def test_tool_result_uses_native_ollama_tool_name():
    message = tool_result_message("current_time", "12:34", tool_call_id="call-1")
    normalized = model_message(message)
    wire = ollama_wire_messages([normalized])[0]
    assert wire == {
        "role": "user",
        "content": "<tool_response>\n12:34\n</tool_response>",
    }
    assert normalized["tool_call_id"] == "call-1"


def test_legacy_tool_name_is_upgraded_for_model_context():
    wire = model_message({"role": "tool", "name": "host_snapshot", "content": "ok"})
    assert wire == {"role": "tool", "tool_name": "host_snapshot", "content": "ok"}


def test_wire_messages_merge_all_system_blocks_into_one_leading_message():
    wire = ollama_wire_messages([
        {"role": "system", "content": "base policy"},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "working state"},
        {"role": "user", "content": "evidence"},
    ])
    assert [index for index, message in enumerate(wire) if message["role"] == "system"] == [0]
    assert "base policy" in wire[0]["content"]
    assert "working state" in wire[0]["content"]
    assert [message["content"] for message in wire[1:]] == ["hello", "evidence"]


def test_system_canonicalization_preserves_non_system_transaction_order():
    messages = canonicalize_system_messages([
        {"role": "system", "content": "base"},
        {"role": "assistant", "content": "", "tool_calls": [_call("demo", 1, "c1")]},
        {"role": "system", "content": "state"},
        {"role": "tool", "tool_name": "demo", "content": "ok", "tool_call_id": "c1"},
    ])
    assert [message["role"] for message in messages] == ["system", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["id"] == "c1"
    assert messages[2]["tool_call_id"] == "c1"


def test_transport_retry_only_happens_before_first_chunk():
    attempts = {"count": 0}
    sleeps = []

    def factory():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionError("cold socket")
        return iter(["a", "b"])

    assert list(stream_with_preflight_retry(factory, retries=1, base_delay=0.1, sleep_fn=sleeps.append)) == ["a", "b"]
    assert attempts["count"] == 2
    assert sleeps == [0.1]


def test_transport_failure_after_first_chunk_is_not_replayed():
    attempts = {"count": 0}

    def factory():
        attempts["count"] += 1
        def stream():
            yield "a"
            raise ConnectionError("stream broke")
        return stream()

    stream = stream_with_preflight_retry(factory, retries=3, base_delay=0, sleep_fn=lambda _: None)
    assert next(stream) == "a"
    try:
        next(stream)
    except ConnectionError:
        pass
    else:
        raise AssertionError("expected post-first-chunk error to propagate")
    assert attempts["count"] == 1


def test_transport_retry_rejects_deterministic_client_errors():
    class ResponseError(Exception):
        status_code = 400

    attempts = {"count": 0}

    def factory():
        attempts["count"] += 1
        raise ResponseError("bad schema")

    try:
        list(stream_with_preflight_retry(factory, retries=3, base_delay=0, sleep_fn=lambda _: None))
    except ResponseError:
        pass
    else:
        raise AssertionError("expected deterministic 4xx error to propagate")
    assert attempts["count"] == 1


def test_retryable_transport_error_classifies_server_and_rate_limit_errors():
    class ResponseError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    assert is_retryable_transport_error(ConnectionError("offline"))
    assert is_retryable_transport_error(ResponseError(429))
    assert is_retryable_transport_error(ResponseError(503))
    assert not is_retryable_transport_error(ResponseError(404))


def test_prompt_template_errors_are_deterministic_even_when_server_returns_500():
    class ResponseError(Exception):
        status_code = 500

    exc = ResponseError("Jinja Exception: System message must be at the beginning.")
    assert is_prompt_protocol_error(exc)
    assert not is_retryable_transport_error(exc)

    attempts = {"count": 0}

    def factory():
        attempts["count"] += 1
        raise exc

    try:
        list(stream_with_preflight_retry(factory, retries=3, base_delay=0, sleep_fn=lambda _: None))
    except ResponseError:
        pass
    else:
        raise AssertionError("expected deterministic prompt protocol error to propagate")
    assert attempts["count"] == 1




def test_single_dict_tool_call_is_not_iterated_as_mapping_keys():
    call = _call("demo", 7, "single")
    calls = merge_stream_tool_calls([], call)
    assert calls == [call]


def test_consume_chat_stream_merges_tools_and_streams_guarded_content():
    calls = [_call("first", 1, "a"), _call("second", 2, "b")]
    chunks = [
        {"message": {"thinking": "plan", "content": "", "tool_calls": [calls[0]]}},
        {"message": {"content": "hello ", "tool_calls": [calls[1]]}},
        {"message": {"content": "world\nthis is visible", "tool_calls": []}, "done": True},
    ]
    thinking = []
    visible = []
    ticks = iter([1.0, 2.0, 3.0, 4.0])

    capture = consume_chat_stream(
        chunks,
        content_stream_allowed=True,
        leak_detector=lambda _text: False,
        on_thinking=thinking.append,
        on_visible_content=visible.append,
        now=lambda: next(ticks),
        guard_chars=999,
        guard_line_chars=8,
    )

    assert capture.content == "hello world\nthis is visible"
    assert [item["function"]["name"] for item in capture.tool_calls] == ["first", "second"]
    assert thinking == ["plan"]
    assert visible == ["hello world\nthis is visible"]
    assert capture.first_token_at == 1.0
    assert capture.first_visible_at == 2.0
    assert capture.cancelled is False


def test_consume_chat_stream_suppresses_policy_leak_before_release():
    visible = []
    capture = consume_chat_stream(
        [{"message": {"content": "SECRET POLICY text that must not stream"}}],
        content_stream_allowed=True,
        leak_detector=lambda text: "SECRET POLICY" in text,
        on_visible_content=visible.append,
        guard_chars=8,
    )

    assert capture.policy_leak_detected is True
    assert capture.content.startswith("SECRET POLICY")
    assert visible == []
    assert capture.first_visible_at is None


def test_consume_chat_stream_honors_cancellation_without_draining_stream():
    consumed = []

    def chunks():
        for value in ("one", "two", "three"):
            consumed.append(value)
            yield {"message": {"content": value}}

    checks = iter([False, True])
    capture = consume_chat_stream(
        chunks(),
        content_stream_allowed=False,
        leak_detector=lambda _text: False,
        cancel_requested=lambda: next(checks),
    )

    assert capture.cancelled is True
    assert capture.content == "one"
    assert consumed == ["one", "two"]


def test_qwen_xml_tool_call_parser_handles_multiline_and_multiple_calls():
    from al_agent.model_protocol import extract_qwen_xml_tool_calls

    text = """Planning first.\n<tool_call>\n<function=web_search>\n<parameter=query>\nLondon Ontario weather\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read_lines>\n<parameter=path>notes.txt</parameter>\n<parameter=start_line>10</parameter>\n<parameter=end_line>20</parameter>\n</function>\n</tool_call>"""
    calls, errors = extract_qwen_xml_tool_calls(text)
    assert errors == []
    assert [c["function"]["name"] for c in calls] == ["web_search", "read_lines"]
    assert calls[0]["function"]["arguments"] == {"query": "London Ontario weather"}
    assert calls[1]["function"]["arguments"]["start_line"] == "10"


def test_qwen_xml_parser_preserves_multiline_parameter_content_and_rejects_suffix():
    from al_agent.model_protocol import extract_qwen_xml_tool_calls

    text = """<tool_call>\n<function=write_file>\n<parameter=content>\nline one\n  line two\n</parameter>\n</function>\n</tool_call>"""
    calls, errors = extract_qwen_xml_tool_calls(text)
    assert errors == []
    assert calls[0]["function"]["arguments"]["content"] == "line one\n  line two"

    calls, errors = extract_qwen_xml_tool_calls(text + "\nnot allowed after tool call")
    assert calls == []
    assert errors == ["unexpected text after final Qwen XML tool call"]
