from __future__ import annotations

from al_agent.micro_model import MicroValidator, validate_completion_with_model


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"response": item}


def test_micro_completion_true_is_structured_and_terminal():
    client = FakeClient(['{"complete":true}'])
    report = validate_completion_with_model(
        client,
        "agent-micro:0.8b",
        "Check host health",
        "host_snapshot ok",
        options={"num_ctx": 2048, "num_predict": 32},
        keep_alive="2m",
    )
    assert report is not None
    assert report["decision"] == "finish"
    assert report["diagnosis"] == "task_complete"
    assert report["validator"] == "micro_model"
    call = client.calls[0]
    assert call["think"] is False
    assert call["format"]["additionalProperties"] is False
    assert call["format"]["required"] == ["complete"]


def test_micro_completion_false_always_falls_through_to_fast_model():
    client = FakeClient(['{"complete":false}'])
    assert validate_completion_with_model(client, "micro", "request", "more work needed") is None


def test_micro_malformed_output_fails_open_to_fast_model():
    client = FakeClient(["not json"])
    assert validate_completion_with_model(client, "micro", "request", "state") is None


def test_micro_transport_error_fails_open_to_fast_model():
    client = FakeClient([RuntimeError("ollama unavailable")])
    assert validate_completion_with_model(client, "micro", "request", "state") is None


def test_micro_validator_can_be_disabled():
    validator = MicroValidator(FakeClient([]), {"enabled": False})
    assert validator.validate_loop("request", "state") is None
    assert validator.health()["enabled"] is False
