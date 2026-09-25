from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_system1_router import FakeRouterClient, make_router, schema

from al_agent.model_traces import router_trace_callback
from al_agent.router_warmup import RouterPrefixWarmer
from al_agent.routing_index import RoutingIndexTooLarge, build_routing_index

CATALOG = [
    schema("current_time", "Read the clock in a timezone"),
    schema("weather_forecast", "Read the forecast for a location"),
]


def test_real_warmup_and_unrelated_queries_share_identical_catalog_prefix(tmp_path):
    client = FakeRouterClient("001H")
    router = make_router(tmp_path, client)
    assert router.warm(CATALOG)
    warm_call = client.calls[-1]
    router.decide("Check the current local time", CATALOG)
    router.decide("What is the weather forecast?", list(reversed(CATALOG)))
    prefix = router.build_index(CATALOG).prefix
    assert all(call["prompt"].startswith(prefix) for call in client.calls)
    assert len({call["prompt"] for call in client.calls}) == 3
    assert "Choices|000\n" in warm_call["prompt"]
    assert all(call["options"] == warm_call["options"] for call in client.calls)
    assert all(call["keep_alive"] == -1 for call in client.calls)
    router.reset_turn()
    assert router._last_prompt == ""
    assert router.build_index(CATALOG).prefix == prefix


def test_index_tracks_schema_changes_and_registry_order_is_irrelevant():
    first = build_routing_index(CATALOG)
    reordered = build_routing_index(reversed(CATALOG))
    changed = build_routing_index(
        [schema("current_time", "Read the clock in a timezone", ["zone"]), CATALOG[1]]
    )
    assert first == reordered
    assert first.fingerprint != changed.fingerprint
    assert first.ids["current_time"] == "001"


def test_default_index_contains_entire_builtin_catalog_without_schemas():
    from tools.builtin_manifest import BUILTIN_MANIFEST

    schemas = [entry["schema"] for entry in BUILTIN_MANIFEST]
    index = build_routing_index(schemas)
    names = {s["function"]["name"] for s in schemas} - {"tool_search", "load_tools"}
    assert set(index.names) == names
    assert all("|" + name + "|" in index.prefix for name in names)
    assert len(index.prefix.encode()) <= (8192 - 1024) * 2
    assert '"parameters"' not in index.prefix


def test_oversized_index_falls_back_without_silent_partial_catalog(tmp_path):
    with pytest.raises(RoutingIndexTooLarge):
        build_routing_index(CATALOG, max_prefix_bytes=10)
    client = FakeRouterClient()
    router = make_router(tmp_path, client)
    router.prefix_max_bytes = 10
    result = router.decide("current time", CATALOG)
    assert result.selected == () and result.router_error
    assert not client.calls
    assert result.candidates  # tool_search can still return compact discovery.


@pytest.mark.parametrize("reply", ["003H", "weather_forecast", "001H extra", "000L"])
def test_malformed_or_unavailable_choices_never_activate_a_schema(tmp_path, reply):
    router = make_router(tmp_path, FakeRouterClient(reply))
    assert router.decide("current time", CATALOG).selected == ()


def test_valid_catalog_id_outside_shortlist_is_rejected(tmp_path):
    client = FakeRouterClient("003H")
    router = make_router(tmp_path, client)
    router.candidate_count = 2
    catalog = [
        schema("clock", "current time"),
        schema("current_time", "current time"),
        schema("weather", "rain forecast"),
    ]
    decision = router.decide("current time", catalog)
    assert len(decision.candidates) == 2
    assert "003" not in client.calls[-1]["prompt"].split("Choices|")[1].splitlines()[0]
    assert not decision.selected


def test_ollama_metrics_and_generate_request_are_written_to_trace(tmp_path):
    class Measured(FakeRouterClient):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                response="001H",
                total_duration=200_000_000,
                load_duration=1_000_000,
                prompt_eval_count=120,
                prompt_eval_cached_count=100,
                prompt_eval_duration=50_000_000,
                eval_count=2,
                eval_duration=140_000_000,
            )

    router = make_router(tmp_path, Measured())
    path = tmp_path / "traces.jsonl"
    router.on_metrics = router_trace_callback(
        path=str(path),
        enabled=True,
        max_bytes=1048576,
        model=router.model,
        options=router.options,
        conversation_id="example",
        turn_id=12,
    )
    decision = router.decide("current time", CATALOG)
    record = json.loads(path.read_text())
    assert decision.selected == ("current_time",)
    assert record["purpose"] == "tool_routing"
    assert record["turn_id"] == 12
    assert record["request"]["api"] == "generate"
    assert record["request"]["prompt"].startswith(router.build_index(CATALOG).prefix)
    assert record["metrics"]["prompt_eval_cached_count"] == 100
    assert record["metrics"]["prompt_eval_duration"] == 50_000_000
    assert record["metrics"]["wall_ms"] >= record["metrics"]["queue_wait_ms"]


def test_old_servers_and_transport_errors_preserve_unknown_metrics(tmp_path):
    router = make_router(tmp_path)
    decision = router.decide("current time", CATALOG)
    assert decision.metrics["prompt_eval_cached_count"] is None
    events = []
    router.on_metrics = events.append

    def broken(**kwargs):
        raise TimeoutError("unavailable")

    router.client.generate = broken
    decision = router.decide("current time", CATALOG)
    assert decision.router_error and not decision.selected
    assert events[-1]["error"] == "TimeoutError: unavailable"
    assert events[-1]["metrics"]["wall_ms"] >= 0
    assert events[-1]["metrics"]["prompt_eval_duration"] is None


def make_warmer(tmp_path):
    router = make_router(tmp_path)
    clock = [0.0]
    catalog = list(CATALOG)
    models = [{"name": router.model, "digest": "v1", "context_length": 8192}]
    warmer = RouterPrefixWarmer(
        router,
        catalog=lambda: catalog,
        resident_models=lambda: {"models": list(models)},
        clock=lambda: clock[0],
        retry_seconds=10,
    )
    return warmer, router, clock, catalog, models


def test_resident_router_gets_no_repeated_keepalive_inference(tmp_path):
    warmer, router, clock, _, _ = make_warmer(tmp_path)
    assert warmer.tick()
    for tick in range(1, 20):
        clock[0] = tick * 30
        assert not warmer.tick()
    assert len(router.client.calls) == 1
    assert warmer.status["state"] == "resident"


@pytest.mark.parametrize(
    "change", ["missing", "digest", "context", "catalog", "options"]
)
def test_changed_catalog_or_model_state_triggers_one_reprime(tmp_path, change):
    warmer, router, clock, catalog, models = make_warmer(tmp_path)
    assert warmer.tick()
    clock[0] = 30
    if change == "missing":
        models.clear()
    elif change == "digest":
        models[0]["digest"] = "v2"
    elif change == "context":
        models[0]["context_length"] = 2048
    elif change == "catalog":
        catalog.append(schema("files", "List files"))
    else:
        router.options["temperature"] = 0.1
    assert warmer.tick()
    assert len(router.client.calls) == 2
    assert not warmer.tick()  # repeated eviction is throttled too


def test_busy_slot_defers_warmup_without_losing_the_retry(tmp_path):
    warmer, router, _, _, _ = make_warmer(tmp_path)

    @contextmanager
    def busy():
        raise BlockingIOError()
        yield

    warmer.inference_slot = busy
    assert not warmer.tick()
    assert warmer.status["state"] == "deferred"
    assert not router.client.calls
    from contextlib import nullcontext

    warmer.inference_slot = nullcontext
    assert warmer.tick()


def test_failed_warmup_backs_off_and_then_recovers(tmp_path):
    warmer, router, clock, _, _ = make_warmer(tmp_path)
    original = router.client.generate

    def fail(**kwargs):
        router.client.calls.append(kwargs)
        raise TimeoutError("offline")

    router.client.generate = fail
    assert not warmer.tick()
    assert not warmer.tick()
    assert len(router.client.calls) == 1
    clock[0] = 30
    router.client.generate = original
    assert warmer.tick()
    assert len(router.client.calls) == 2


def test_server_outage_invalidates_prior_warmup(tmp_path):
    warmer, router, clock, _, _ = make_warmer(tmp_path)
    assert warmer.tick()
    poll = warmer.resident_models

    def offline():
        raise ConnectionError("server restarting")

    warmer.resident_models = offline
    assert not warmer.tick()
    assert warmer.status["state"] == "check_failed"
    warmer.resident_models = poll
    clock[0] = 30
    assert warmer.tick()
    assert len(router.client.calls) == 2


def test_runtime_passes_routed_schema_to_main_and_records_router_metrics(
    tmp_path, monkeypatch
):
    from al_agent import runtime, state
    from tools.catalog import catalog_snapshot
    from tools.conversation_context import conversation_context
    from tools.memory import _load_chat_history_from_db

    schemas, _, _ = catalog_snapshot()
    index = build_routing_index(schemas)
    router_client = FakeRouterClient(index.ids["calculate"] + "H")

    class MainClient:
        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            content = (
                "<tool_call>\n<function=calculate>\n<parameter=expression>\n6 * 7\n"
                "</parameter>\n</function>\n</tool_call>"
                if len(self.requests) == 1
                else "42"
            )
            return iter(
                [{"message": {"role": "assistant", "content": content}, "done": True}]
            )

    main = MainClient()
    trace_path = tmp_path / "routing-trace.jsonl"
    monkeypatch.setattr(runtime, "OLLAMA", main)
    monkeypatch.setattr(runtime, "ROUTER_OLLAMA", router_client)
    monkeypatch.setattr(state, "MODEL_TRACE_ENABLED", True)
    monkeypatch.setattr(state, "MODEL_TRACE_PATH", str(trace_path))
    monkeypatch.setitem(state.AGENT_CFG, "tool_protocol", "qwen_xml")
    with conversation_context("router-integration-" + tmp_path.name):
        runtime.handle_user_turn([], "Calculate 6 times 7", False, refresh_history=True)
        assert _load_chat_history_from_db()[-1]["content"] == "42"
    assert len(router_client.calls) == 1
    assert len(main.requests) == 2
    assert {s["function"]["name"] for s in main.requests[0]["tools"]} == {
        "tool_search",
        "load_tools",
        "calculate",
    }
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    routes = [r for r in traces if r["purpose"] == "tool_routing"]
    assert len(routes) == 1
    assert routes[0]["metrics"]["prefix_fingerprint"] == index.fingerprint
    assert routes[0]["metrics"]["queue_wait_ms"] >= 0
