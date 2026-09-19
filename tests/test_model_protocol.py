from al_agent.model_protocol import (
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
    assert wire["tool_name"] == "current_time"
    assert "name" not in wire
    assert "tool_call_id" not in wire
    assert normalized["tool_call_id"] == "call-1"


def test_legacy_tool_name_is_upgraded_for_model_context():
    wire = model_message({"role": "tool", "name": "host_snapshot", "content": "ok"})
    assert wire == {"role": "tool", "tool_name": "host_snapshot", "content": "ok"}


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
