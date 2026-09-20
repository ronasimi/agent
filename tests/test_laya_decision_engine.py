from __future__ import annotations

import sys
import types

from al_agent.decision_engine import (
    DecisionEngineClient, LocalLayaEngine, recommended_tools_for_family, tool_family_for_name,
)


class _FakeAgent:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {name: dict(self.answers.get(name, {})) for name in questions}}


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
    def json(self):
        return self._payload


def _client(monkeypatch, answers):
    def post(url, json=None, timeout=None):
        return _Response({"ok": True, "trace_id": "trace", "latency_ms": 12.5, "answers": {
            name: dict(answers.get(name, {})) for name in (json or {}).get("questions", {})
        }})
    monkeypatch.setattr("al_agent.decision_engine.requests.post", post)
    return DecisionEngineClient({
        "enabled": True,
        "endpoint": "http://127.0.0.1:8091",
        "training_capture": {"enabled": False},
        "routing": {"enabled": True, "thresholds": {}},
        "validator": {"enabled": True, "min_confidence": 0.95},
    })


def test_route_turn_batches_decisions(monkeypatch):
    client = _client(monkeypatch, {
        "route_family": {"choice": "network", "confidence": 0.98},
        "tool_requirement": {"choice": "required", "confidence": 0.99},
        "freshness": {"choice": "current", "confidence": 0.97},
        "continuation": {"choice": "new", "confidence": 0.99},
        "renderer": {"choice": "model", "confidence": 0.96},
        "risk": {"choice": "read_only", "confidence": 0.99},
    })
    result = client.route_turn("Diagnose why example.com does not resolve")
    assert result.ok
    assert result.accepted("route_family", 0.92).value == "network"
    assert result.latency_ms == 12.5


def test_low_confidence_is_not_accepted(monkeypatch):
    client = _client(monkeypatch, {"route_family": {"choice": "network", "confidence": 0.51}})
    result = client.route_turn("ambiguous thing")
    assert result.ok
    assert result.accepted("route_family", 0.92) is None


def test_validator_only_bypasses_fast_model_for_terminal_high_confidence(monkeypatch):
    client = _client(monkeypatch, {
        "action": {"choice": "finish", "confidence": 0.98},
        "diagnosis": {"choice": "task_complete", "confidence": 0.96},
    })
    report = client.validate_loop("check host", "host_snapshot ok", candidate_tools=["host_snapshot"])
    assert report["decision"] == "finish"
    assert report["validator"] == "laya"

    client2 = _client(monkeypatch, {
        "action": {"choice": "recover", "confidence": 0.99},
        "diagnosis": {"choice": "wrong_tool", "confidence": 0.99},
    })
    assert client2.validate_loop("check host", "bad result", candidate_tools=["host_snapshot"]) is None


def test_family_recommendations_never_grant_mutating_execution_tools():
    all_recommended = {name for family in ("system", "network", "files", "web", "memory", "coding", "workflow", "research", "automation") for name in recommended_tools_for_family(family)}
    assert "execute_shell" not in all_recommended
    assert "execute_python" not in all_recommended
    assert "write_file" not in all_recommended
    assert tool_family_for_name("dns_diagnose") == "network"
    assert tool_family_for_name("repo_checks") == "coding"


def test_local_engine_load_and_single_forward(monkeypatch):
    fake = _FakeAgent({"x": {"choice": "yes", "confidence": 0.99}})
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=lambda *a, **k: fake))
    engine = LocalLayaEngine({"model": "fake", "device": "cpu", "max_state_chars": 100})
    assert engine.load()
    result = engine.predict({"body": "x" * 1000}, {"x": {"type": "choice", "criteria": {"yes": "yes", "no": "no"}}})
    assert result["answers"]["x"]["choice"] == "yes"
    assert len(fake.calls) == 1
    assert len(fake.calls[0][0]["body"]) <= engine.max_state_chars
    assert engine.unload()


def test_fast_model_route_fallback_is_constrained_and_labeled():
    from al_agent.decision_engine import route_with_fast_model

    class FastClient:
        def generate(self, **kwargs):
            assert kwargs["think"] is False
            assert kwargs["format"]["additionalProperties"] is False
            return {"response": '{"route_family":"network","tool_requirement":"required","freshness":"current","continuation":"new","renderer":"model","risk":"read_only"}'}

    batch = route_with_fast_model(
        FastClient(), "fast:2b", "Diagnose DNS for example.com",
        options={"num_ctx": 2048}, keep_alive="2m",
    )
    assert batch.ok
    assert batch.source == "fast_model"
    assert batch.accepted("route_family", 0.99).value == "network"
    assert batch.accepted("tool_requirement", 0.99).value == "required"


def test_fast_model_route_fallback_rejects_invalid_labels():
    from al_agent.decision_engine import route_with_fast_model

    class FastClient:
        def generate(self, **kwargs):
            return {"response": '{"route_family":"sudo_everything","tool_requirement":"required","freshness":"current","continuation":"new","renderer":"model","risk":"mutation"}'}

    batch = route_with_fast_model(FastClient(), "fast:2b", "ambiguous")
    assert not batch.ok
    assert "invalid route_family" in batch.error


def test_local_engine_unload_cancels_inflight_cold_load(monkeypatch):
    import threading

    started = threading.Event()
    release = threading.Event()
    fake = _FakeAgent({"x": {"choice": "yes", "confidence": 0.99}})

    def slow_load(*args, **kwargs):
        started.set()
        release.wait(timeout=2)
        return fake

    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(load=slow_load))
    engine = LocalLayaEngine({"model": "fake", "device": "cpu"})
    thread = threading.Thread(target=engine.load)
    thread.start()
    assert started.wait(timeout=1)
    engine.unload()
    release.set()
    thread.join(timeout=2)
    assert not engine.loaded
