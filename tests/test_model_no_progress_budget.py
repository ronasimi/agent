def test_empty_main_responses_stop_before_global_model_call_budget(monkeypatch):
    from al_agent import turn_engine as te

    class EmptyClient:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return iter([{
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 1,
            }])

    client = EmptyClient()
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_NO_PROGRESS_MAX_RETRIES", 2)
    monkeypatch.setattr(te, "MAX_MODEL_CALLS_PER_TURN", 6)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages, "when is the next full moon?", False,
        runtime_overrides={
            "OLLAMA": client,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    assert client.calls == 2
    content = messages[-1]["content"]
    assert "stopped before the global model-call budget was exhausted" in content
    assert "hard turn/model-call budget exhausted" not in content


def test_thinking_only_response_gets_one_bounded_conclusion_retry(monkeypatch):
    from al_agent import turn_engine as te

    class ThinkingThenContentClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return iter([{
                    "message": {"thinking": "internal reasoning only", "content": ""},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 1,
                    "eval_count": 384,
                }])
            return iter([{
                "message": {"thinking": "brief", "content": "Hello!"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 3,
            }])

    client = ThinkingThenContentClient()
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_NO_PROGRESS_MAX_RETRIES", 2)
    monkeypatch.setattr(te, "MAX_MODEL_CALLS_PER_TURN", 6)
    monkeypatch.setattr(te, "REASONING_RECOVERY_ENABLED", True)
    monkeypatch.setattr(te, "REASONING_RECOVERY_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(te, "REASONING_RECOVERY_THINK_MODE", "low")
    monkeypatch.setattr(te, "REASONING_RECOVERY_FINAL_NUM_PREDICT", 2048)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages, "Hello?", False,
        runtime_overrides={
            "OLLAMA": client,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    assert len(client.calls) == 2
    assert client.calls[0]["think"] is False
    assert client.calls[1]["think"] == "low"
    assert client.calls[1]["options"]["num_predict"] >= 2048
    assert messages[-1]["content"] == "Hello!"


def test_repeated_thinking_only_responses_report_specific_failure(monkeypatch):
    from al_agent import turn_engine as te

    class ThinkingOnlyClient:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return iter([{
                "message": {"thinking": "internal reasoning only", "content": ""},
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 1,
                "eval_count": kwargs.get("options", {}).get("num_predict", 1),
            }])

    client = ThinkingOnlyClient()
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_NO_PROGRESS_MAX_RETRIES", 2)
    monkeypatch.setattr(te, "MAX_MODEL_CALLS_PER_TURN", 6)
    monkeypatch.setattr(te, "REASONING_RECOVERY_ENABLED", True)
    monkeypatch.setattr(te, "REASONING_RECOVERY_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages, "Hello?", False,
        runtime_overrides={
            "OLLAMA": client,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: object(),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    assert client.calls == 2
    assert "internal reasoning without transitioning" in messages[-1]["content"]
