from __future__ import annotations

from al_agent.vision import has_images, route_multimodal_messages


class FakeClient:
    def __init__(self, content: str = "A terminal window shows a Python traceback."):
        self.content = content
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.content}}


def test_same_model_vision_role_is_single_pass_and_keeps_images():
    client = FakeClient()
    messages = [{"role": "user", "content": "What is shown?", "images": ["abc123"]}]

    route = route_multimodal_messages(
        client,
        messages,
        main_model="agent-main:2b",
        main_options={"num_ctx": 8192, "temperature": 0.2},
        vision_model="agent-main:2b",
        vision_options={"num_ctx": 8192, "temperature": 0.2},
        vision_keep_alive=-1,
    )

    assert route.model == "agent-main:2b"
    assert route.options["num_ctx"] == 8192
    assert route.keep_alive == -1
    assert route.used_sidecar is False
    assert has_images(route.messages)
    assert client.calls == []  # no duplicate vision inference when roles share a runner


def test_distinct_vision_model_becomes_no_tools_sidecar_and_main_gets_text_only():
    client = FakeClient("A router admin page is visible with a red error banner.")
    cache: dict[str, str] = {}
    messages = [{"role": "user", "content": "Diagnose this screenshot", "images": ["abc123"]}]

    route = route_multimodal_messages(
        client,
        messages,
        main_model="agent-main:2b",
        main_options={"num_ctx": 8192},
        vision_model="vision:latest",
        vision_options={"num_ctx": 4096},
        vision_keep_alive="2m",
        cache=cache,
    )

    assert route.model == "agent-main:2b"
    assert route.used_sidecar is True
    assert not has_images(route.messages)
    assert "Harness vision observation" in route.messages[0]["content"]
    assert "red error banner" in route.messages[0]["content"]
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "vision:latest"
    assert call["tools"] == []
    assert call["think"] is False
    assert call["stream"] is False
    assert call["messages"][0]["images"] == ["abc123"]

    # Rebuilding the prompt during a tool loop must reuse the visual observation.
    route_multimodal_messages(
        client,
        messages,
        main_model="agent-main:2b",
        main_options={"num_ctx": 8192},
        vision_model="vision:latest",
        vision_options={"num_ctx": 4096},
        vision_keep_alive="2m",
        cache=cache,
    )
    assert len(client.calls) == 1


def test_non_image_turn_always_uses_main_role():
    client = FakeClient()
    route = route_multimodal_messages(
        client,
        [{"role": "user", "content": "hello"}],
        main_model="agent-main:2b",
        main_options={"num_ctx": 8192},
        vision_model="vision:latest",
        vision_options={"num_ctx": 4096},
        vision_keep_alive="2m",
    )
    assert route.model == "agent-main:2b"
    assert route.used_sidecar is False
    assert client.calls == []
