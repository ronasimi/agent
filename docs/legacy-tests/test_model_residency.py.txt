from contextlib import nullcontext


def test_report_stage_swaps_model_roles_and_restores_executor_plus_decision(monkeypatch):
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
    monkeypatch.setattr(residency, "MODEL", "executor:2b")
    monkeypatch.setattr(residency, "FAST_MODEL", "executor:2b")
    monkeypatch.setattr(residency, "DECISION_MODEL", "decision:0.8b")
    monkeypatch.setattr(residency, "REASONING_MODEL", "reasoning:4b")
    monkeypatch.setattr(residency, "VISION_MODEL", "reasoning:4b")
    monkeypatch.setattr(residency, "REPORT_MODEL", "report:9b")
    monkeypatch.setattr(residency, "REPORT_MODEL_KEEP_ALIVE", -1)
    monkeypatch.setattr(residency, "FAST_MODEL_KEEP_ALIVE", -1)
    monkeypatch.setattr(residency, "REPORT_RESTORE_MODELS", True)
    fast_scheduled = []
    decision_scheduled = []
    monkeypatch.setattr(residency, "schedule_fast_model_prewarm", lambda reason="": fast_scheduled.append(reason) or True)
    monkeypatch.setattr(residency, "schedule_decision_model_prewarm", lambda reason="": decision_scheduled.append(reason) or True)

    residency.enter_report_model_stage("job-1")
    residency.exit_report_model_stage("job-1", restore=True)

    assert calls[:4] == [
        ("executor:2b", 0, False),
        ("decision:0.8b", 0, False),
        ("reasoning:4b", 0, False),
        ("report:9b", -1, True),
    ]
    assert calls[4:] == [
        ("report:9b", 0, False),
        ("executor:2b", -1, True),
    ]
    assert fast_scheduled == ["post-report"]
    assert decision_scheduled == ["post-report"]
    assert any(key == "agent.report_model_active" and value is False for key, value in state_updates)


def test_reasoning_escalation_evicts_decision_runner_not_executor(monkeypatch):
    from al_agent import model_residency as residency

    calls = []

    class FakeClient:
        def ps(self):
            return {"models": [
                {"model": "executor:2b", "context_length": 16384},
                {"model": "decision:0.8b", "context_length": 8192},
            ]}
        def chat(self, **kwargs):
            calls.append((kwargs.get("model"), kwargs.get("keep_alive")))
            return {"message": {"content": ""}}

    monkeypatch.setattr(residency, "_client", lambda: FakeClient())
    monkeypatch.setattr(residency, "MODEL", "executor:2b")
    monkeypatch.setattr(residency, "DECISION_MODEL", "decision:0.8b")
    monkeypatch.setattr(residency, "REASONING_MODEL", "reasoning:4b")
    monkeypatch.setattr(residency, "REASONING_OPTIONS", {"num_ctx": 16384})

    assert residency.prepare_reasoning_model_for_interactive() is True
    assert calls == [("decision:0.8b", 0)]


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


def test_decision_inference_evicts_lingering_reasoning_runner(monkeypatch):
    from al_agent import model_residency as residency

    calls = []

    class FakeClient:
        def ps(self):
            return {"models": [
                {"model": "executor:2b", "context_length": 16384},
                {"model": "reasoning:4b", "context_length": 16384},
            ]}

        def chat(self, **kwargs):
            calls.append((kwargs.get("model"), kwargs.get("keep_alive")))
            return {"message": {"content": ""}}

    monkeypatch.setattr(residency, "_client", lambda: FakeClient())
    monkeypatch.setattr(residency, "MODEL", "executor:2b")
    monkeypatch.setattr(residency, "DECISION_MODEL", "decision:0.8b")
    monkeypatch.setattr(residency, "DECISION_OPTIONS", {"num_ctx": 8192})
    monkeypatch.setattr(residency, "REASONING_MODEL", "reasoning:4b")
    monkeypatch.setattr(residency, "REASONING_OPTIONS", {"num_ctx": 16384})

    assert residency.prepare_decision_model_for_interactive() is True
    assert calls == [("reasoning:4b", 0)]


def test_idle_decision_restore_unloads_reasoner_before_warm(monkeypatch):
    from al_agent import model_residency as residency

    calls = []
    state = {"models": [
        {"model": "executor:2b", "context_length": 16384},
        {"model": "reasoning:4b", "context_length": 16384},
    ]}

    class FakeClient:
        def ps(self):
            return {"models": list(state["models"])}

        def chat(self, **kwargs):
            model = kwargs.get("model")
            keep_alive = kwargs.get("keep_alive")
            calls.append((model, keep_alive))
            if keep_alive == 0:
                state["models"] = [row for row in state["models"] if row["model"] != model]
            elif model == "decision:0.8b":
                state["models"].append({"model": model, "context_length": 8192})
            return {"message": {"content": ""}}

    monkeypatch.setattr(residency, "_client", lambda: FakeClient())
    monkeypatch.setattr(residency, "MODEL", "executor:2b")
    monkeypatch.setattr(residency, "DECISION_MODEL", "decision:0.8b")
    monkeypatch.setattr(residency, "DECISION_OPTIONS", {"num_ctx": 8192})
    monkeypatch.setattr(residency, "DECISION_MODEL_KEEP_ALIVE", -1)
    monkeypatch.setattr(residency, "REASONING_MODEL", "reasoning:4b")
    monkeypatch.setattr(residency, "REASONING_OPTIONS", {"num_ctx": 16384})
    monkeypatch.setattr(residency, "model_maintenance_slot", lambda **kwargs: nullcontext())
    monkeypatch.setattr("al_agent.background.resources._interactive_busy", lambda: False)
    monkeypatch.setattr("al_agent.state.MODEL_CAPABILITY_PROBE_DECISION", False)
    monkeypatch.setattr(residency, "_decision_prewarm_thread", object())

    residency._prewarm_decision_when_idle("test")

    assert calls == [("reasoning:4b", 0), ("decision:0.8b", -1)]


def test_vision_request_evicts_decision_and_preserves_executor(monkeypatch):
    from al_agent import model_residency as residency

    calls = []

    class FakeClient:
        def ps(self):
            return {"models": [
                {"model": "agent-main", "context_length": 16384},
                {"model": "agent-micro", "context_length": 8192},
            ]}

        def chat(self, **kwargs):
            calls.append((kwargs.get("model"), kwargs.get("keep_alive")))
            return {"message": {"content": ""}}

    monkeypatch.setattr(residency, "_client", lambda: FakeClient())
    monkeypatch.setattr(residency, "MODEL", "agent-main")
    monkeypatch.setattr(residency, "DECISION_MODEL", "agent-micro")
    monkeypatch.setattr(residency, "VISION_MODEL", "qwen3.5:4b")
    monkeypatch.setattr(residency, "VISION_OPTIONS", {"num_ctx": 16384})
    monkeypatch.setattr(residency, "REASONING_MODEL", "agent-reasoning")

    assert residency.prepare_vision_model_for_interactive() is True
    assert calls == [("agent-micro", 0)]
