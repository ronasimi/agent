from contextlib import nullcontext


def test_report_stage_swaps_models_and_restores_normal_residency(monkeypatch):
    from al_agent import model_residency as residency

    calls = []
    state_updates = []

    class FakeClient:
        def chat(self, **kwargs):
            calls.append((kwargs.get("model"), kwargs.get("keep_alive"), bool(kwargs.get("options"))))
            return {"message": {"content": ""}}

    monkeypatch.setattr(residency, "_client", lambda: FakeClient())
    monkeypatch.setattr(residency, "background_inference_slot", lambda: nullcontext())
    monkeypatch.setattr(residency, "record_monitor_state", lambda key, value: state_updates.append((key, value)))
    monkeypatch.setattr(residency, "MODEL", "main:4b")
    monkeypatch.setattr(residency, "FAST_MODEL", "fast:2b")
    monkeypatch.setattr(residency, "MICRO_MODEL", "micro:0.8b")
    monkeypatch.setattr(residency, "REPORT_MODEL", "report:9b")
    monkeypatch.setattr(residency, "REPORT_MODEL_KEEP_ALIVE", -1)
    monkeypatch.setattr(residency, "FAST_MODEL_KEEP_ALIVE", "2m")
    monkeypatch.setattr(residency, "REPORT_RESTORE_MODELS", True)
    monkeypatch.setattr(residency, "REPORT_RESTORE_FAST_MODEL", True)

    residency.enter_report_model_stage("job-1")
    residency.exit_report_model_stage("job-1", restore=True)

    assert calls[:4] == [
        ("main:4b", 0, False),
        ("fast:2b", 0, False),
        ("micro:0.8b", 0, False),
        ("report:9b", -1, True),
    ]
    assert calls[4:] == [
        ("report:9b", 0, False),
        ("main:4b", -1, True),
        ("fast:2b", "2m", True),
    ]
    assert any(key == "agent.report_model_active" and value is False for key, value in state_updates)


def test_interactive_turn_evicts_lingering_report_model(monkeypatch):
    from al_agent import model_residency as residency

    calls = []

    class FakeClient:
        def chat(self, **kwargs):
            calls.append((kwargs.get("model"), kwargs.get("keep_alive")))
            return {"message": {"content": ""}}

    monkeypatch.setattr(residency, "_client", lambda: FakeClient())
    monkeypatch.setattr(residency, "get_monitor_state", lambda key, default=False: {"model": "report:9b"})
    monkeypatch.setattr(residency, "record_monitor_state", lambda *args, **kwargs: None)
    assert residency.evict_report_model_for_interactive() is True
    assert calls == [("report:9b", 0)]
