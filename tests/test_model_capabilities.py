import json

import pytest

from al_agent import model_capabilities as caps


class FakeResponse:
    def __init__(self, *, content="", thinking="", tool_calls=None):
        self.message = {
            "role": "assistant",
            "content": content,
            "thinking": thinking,
            "tool_calls": list(tool_calls or []),
        }
        self.done = True
        self.done_reason = "stop"


class FakeClient:
    def __init__(self):
        self.chat_calls = []

    def list(self):
        return {"models": [{"model": "demo:latest", "digest": "sha256:abc123"}]}

    def show(self, model):
        assert model == "demo:latest"
        return {"template": "demo-template", "capabilities": ["tools", "thinking"]}

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        messages = kwargs.get("messages") or []
        tools = kwargs.get("tools")
        think = kwargs.get("think", "missing")
        stream = kwargs.get("stream", False)
        if tools:
            return FakeResponse(tool_calls=[{
                "type": "function",
                "function": {"name": "capability_probe_echo", "arguments": {"text": "PING"}},
            }])
        if messages and messages[-1].get("role") == "tool":
            return FakeResponse(content="PONG")
        if stream:
            if think is True:
                return iter([
                    FakeResponse(thinking="plan"),
                    FakeResponse(content="OK"),
                ])
            return iter([FakeResponse(content="OK")])
        return FakeResponse(content="OK")


def test_probe_detects_tools_thinking_and_streaming_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(caps, "_ACTIVE", {})
    client = FakeClient()
    cache = tmp_path / "model_caps.json"
    profile = caps.probe_model_capabilities(
        client, "demo:latest", options={"num_ctx": 4096}, cache_path=str(cache), force=True,
    )
    assert profile.plain_chat is True
    assert profile.content_streaming is True
    assert profile.think_parameter is True
    assert profile.reasoning_streaming is True
    assert profile.tools_parameter is True
    assert profile.tool_call_mode == "native"
    assert profile.tool_result_continuation is True
    assert "tools" in profile.metadata_capabilities
    first_chat_count = len(client.chat_calls)
    assert first_chat_count >= 4

    # Same model digest + probe version must reuse the persistent result without
    # repeating generation calls.
    second = caps.probe_model_capabilities(
        client, "demo:latest", options={"num_ctx": 4096}, cache_path=str(cache), force=False,
    )
    assert second.identity == profile.identity
    assert len(client.chat_calls) == first_chat_count
    payload = json.loads(cache.read_text())
    assert profile.identity in payload["profiles"]


def test_capability_overrides_omit_explicitly_unsupported_optional_fields(monkeypatch):
    profile = caps.ModelCapabilityProfile(
        model="legacy", identity="v1:legacy:x", think_parameter=False,
        tools_parameter=False, tool_call_mode="unsupported",
    )
    monkeypatch.setattr(caps, "_ACTIVE", {"legacy": profile})
    assert caps.capability_chat_overrides("legacy", think=False, tools=[]) == {}
    with pytest.raises(caps.ModelCapabilityError):
        caps.capability_chat_overrides("legacy", think=False, tools=[{"type": "function"}])


@pytest.mark.parametrize("enabled", [False, True])
def test_unknown_profile_preserves_existing_runtime_behavior(monkeypatch, enabled):
    from tools import config
    monkeypatch.setattr(config, "load_config", lambda: {"agent": {"supports_thinking": enabled}})
    monkeypatch.setattr(caps, "_ACTIVE", {})
    assert caps.capability_chat_overrides("new-model", think=True, tools=[]) == ({"think": True} if enabled else {})


def test_behavioral_probe_must_verify_tool_call_before_runtime_uses_tools(monkeypatch):
    profile = caps.ModelCapabilityProfile(
        model="chatty", identity="v2:chatty:x", think_parameter=True,
        tools_parameter=True, tool_call_mode="accepted_unverified",
        tool_result_continuation=None,
    )
    monkeypatch.setattr(caps, "_ACTIVE", {"chatty": profile})
    with pytest.raises(caps.ModelCapabilityError, match="did not emit a verifiable tool call"):
        caps.capability_chat_overrides(
            "chatty", think=False,
            tools=[{"type": "function", "function": {"name": "demo", "parameters": {"type": "object"}}}],
        )


class NoThinkClient(FakeClient):
    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if "think" in kwargs:
            raise RuntimeError("thinking is not supported by this model")
        if kwargs.get("tools"):
            raise RuntimeError("tools are not supported by this model")
        return iter([FakeResponse(content="OK")]) if kwargs.get("stream") else FakeResponse(content="OK")


def test_probe_degrades_cleanly_when_optional_features_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(caps, "_ACTIVE", {})
    profile = caps.probe_model_capabilities(
        NoThinkClient(), "demo:latest", options={}, cache_path=str(tmp_path / "caps.json"), force=True,
    )
    assert profile.plain_chat is True
    assert profile.content_streaming is True
    assert profile.think_parameter is False
    assert profile.tools_parameter is False
    assert profile.tool_call_mode == "unsupported"
    assert profile.tool_result_continuation is None
