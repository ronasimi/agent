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
    monkeypatch.setattr(residency, "model_maintenance_slot", lambda **kwargs: nullcontext())
    monkeypatch.setattr(residency, "record_monitor_state", lambda key, value: state_updates.append((key, value)))
    monkeypatch.setattr(residency, "MODEL", "main:4b")
    monkeypatch.setattr(residency, "FAST_MODEL", "fast:2b")
    monkeypatch.setattr(residency, "VISION_MODEL", "main:4b")
    monkeypatch.setattr(residency, "REPORT_MODEL", "report:9b")
    monkeypatch.setattr(residency, "REPORT_MODEL_KEEP_ALIVE", -1)
    monkeypatch.setattr(residency, "FAST_MODEL_KEEP_ALIVE", -1)
    monkeypatch.setattr(residency, "REPORT_RESTORE_MODELS", True)
    scheduled = []
    monkeypatch.setattr(residency, "schedule_fast_model_prewarm", lambda reason="": scheduled.append(reason) or True)

    residency.enter_report_model_stage("job-1")
    residency.exit_report_model_stage("job-1", restore=True)

    assert calls[:3] == [
        ("main:4b", 0, False),
        ("fast:2b", 0, False),
        ("report:9b", -1, True),
    ]
    assert calls[3:] == [
        ("report:9b", 0, False),
        ("main:4b", -1, True),
    ]
    assert scheduled == ["post-report"]
    assert any(key == "agent.report_model_active" and value is False for key, value in state_updates)


def test_fast_prewarm_skips_matching_resident_runner(monkeypatch):
    from al_agent import model_residency as residency

    calls = []

    class FakeClient:
        def ps(self):
            return {"models": [{"model": "fast:2b", "context_length": 4096}]}

        def chat(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(residency, "_client", lambda: FakeClient())
    monkeypatch.setattr(residency, "FAST_MODEL", "fast:2b")
    monkeypatch.setattr(residency, "FAST_OPTIONS", {"num_ctx": 4096})
    monkeypatch.setattr(residency, "_fast_prewarm_thread", object())
    monkeypatch.setattr(residency, "model_maintenance_slot", lambda **kwargs: nullcontext())
    monkeypatch.setattr("al_agent.background.resources._interactive_busy", lambda: False)
    monkeypatch.setattr("al_agent.state.MODEL_CAPABILITY_PROBE_FAST", False)

    residency._prewarm_fast_when_idle("test")

    assert calls == []


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
