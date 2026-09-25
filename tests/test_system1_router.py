from __future__ import annotations

from al_agent.routing_decision import RoutingFeedbackStore
from al_agent.system1_router import MAX_REQUEST_CHARS, SystemOneRouter
from al_agent.tool_session import ToolSession


def schema(name: str, description: str, required=None):
    required = required or []
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in required},
                "required": required,
                "additionalProperties": False,
            },
        },
    }


class FakeRouterClient:
    def __init__(self, reply="001H"):
        self.reply = reply
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": self.reply}


def make_router(tmp_path, client=None):
    client = client or FakeRouterClient()
    return SystemOneRouter(
        client,
        model="router:0.5b",
        options={"num_ctx": 8192, "temperature": 0, "num_predict": 4},
        feedback=RoutingFeedbackStore(str(tmp_path / "routing.db")),
    )


def test_tiny_router_selects_tool_with_single_stateless_prompt(tmp_path):
    client = FakeRouterClient("001H")
    router = make_router(tmp_path, client)
    schemas = [
        schema("current_time", "Return the current clock/date for an IANA timezone."),
        schema("time_difference", "Return the difference between two datetimes.", ["a", "b"]),
    ]
    decision = router.decide("Check the current local time", schemas)
    assert decision.selected == ("current_time",)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert "messages" not in call
    assert call["options"]["num_predict"] == 4
    assert "current_time" in call["prompt"]
    assert "parameters" not in call["prompt"]


def test_router_prompt_is_bounded_and_reset(tmp_path):
    client = FakeRouterClient("001H")
    router = make_router(tmp_path, client)
    router.decide("x" * (MAX_REQUEST_CHARS * 4) + " current time", [schema("current_time", "clock " * 100)])
    assert len(router._last_request) <= MAX_REQUEST_CHARS + 3
    assert len(router._last_candidates) == 1
    assert len(router._last_prompt) < 1800
    router.reset_turn()
    assert router._last_prompt == ""
    assert router._last_request == ""
    assert router._last_candidates == ()


def test_persistent_feedback_calibrates_new_router_instance(tmp_path):
    store = RoutingFeedbackStore(str(tmp_path / "routing.db"))
    schemas = [schema("current_time", "Return the current clock/date for an IANA timezone.")]
    first = SystemOneRouter(FakeRouterClient("001H"), model="router", feedback=store)
    before = first.decide("current local time", schemas)
    for _ in range(5):
        first.record("current_time", before.context_key, 1.0, event_type="task_success")
    restarted = SystemOneRouter(
        FakeRouterClient("001H"), model="router",
        feedback=RoutingFeedbackStore(str(tmp_path / "routing.db")),
    )
    after = restarted.decide("current local time", schemas)
    assert after.confidence > before.confidence


def test_router_failure_falls_back_without_guessing_schema(tmp_path):
    class Broken:
        def generate(self, **kwargs):
            raise TimeoutError("router timeout")

    router = make_router(tmp_path, Broken())
    schemas = [schema("current_time", "Return the current clock/date for an IANA timezone.")]
    decision = router.decide("current local time", schemas)
    assert decision.selected == ()
    assert decision.router_error


def test_tool_search_returns_compact_metadata_and_only_selected_schema(tmp_path):
    router = make_router(tmp_path, FakeRouterClient("001H"))
    schemas = [
        schema("current_time", "Return the current clock/date for an IANA timezone."),
        schema("time_difference", "Return the difference between two datetimes.", ["a", "b"]),
    ]
    session = ToolSession(schemas, lambda *a: None, router=router)
    result = session.invoke("tool_search", {"query": "current time", "limit": 2})
    assert result["selected"] == "current_time"
    assert "schemas" not in result
    assert all("function" not in row for row in result["candidates"])
    assert [s["function"]["name"] for s in session.schemas] == [
        "tool_search", "load_tools", "current_time"
    ]


def test_low_confidence_search_clears_previous_task_schema(tmp_path):
    router = make_router(tmp_path, FakeRouterClient("001L"))
    schemas = [
        schema("current_time", "Return the current clock/date for an IANA timezone."),
        schema("time_difference", "Return the difference between two datetimes.", ["a", "b"]),
    ]
    session = ToolSession(schemas, lambda *a: None, router=router)
    session.invoke("load_tools", {"names": ["time_difference"]})
    assert "time_difference" in session.active
    result = session.invoke("tool_search", {"query": "current time", "limit": 2})
    assert result["selected"] is None
    assert session.active == {}
    assert [s["function"]["name"] for s in session.schemas] == ["tool_search", "load_tools"]
