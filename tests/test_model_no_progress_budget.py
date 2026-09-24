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
    assert client.calls[1]["think"] is False
    assert client.calls[1]["options"]["num_predict"] >= 2048
    assert messages[-1]["content"] == "Hello!"


def test_thinking_enabled_recovery_forces_no_think_conclusion(monkeypatch):
    from al_agent import turn_engine as te

    class ThinkingThenContentClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return iter([{
                    "message": {"thinking": "long reasoning", "content": ""},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 1,
                    "eval_count": 1024,
                }])
            return iter([{
                "message": {"content": "Hello!"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 2,
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
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages, "Hello?", True,
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
    assert client.calls[0]["think"] is True
    assert client.calls[1]["think"] is False
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


def _configure_zero_tool_turn_test(monkeypatch, te):
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


def _zero_tool_runtime(client):
    return {
        "OLLAMA": client,
        "record_monitor_state": lambda *a, **k: None,
        "append_and_save": lambda rows, item: rows.append(item),
        "acquire_turn_lock": lambda: object(),
        "release_turn_lock": lambda _lock: None,
        "acquire_inference_lock": lambda: object(),
        "release_inference_lock": lambda _lock: None,
        "queue_compaction_if_needed": lambda *a, **k: None,
    }


def test_zero_tool_invalid_call_gets_direct_no_think_recovery(monkeypatch):
    from al_agent import turn_engine as te

    class InvalidThenContentClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return iter([{
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "ghost_tool", "arguments": {}}}],
                    },
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 4,
                }])
            return iter([{
                "message": {"content": "Hello! How can I help?"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 7,
            }])

    client = InvalidThenContentClient()
    _configure_zero_tool_turn_test(monkeypatch, te)
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(messages, "Hello?", True, runtime_overrides=_zero_tool_runtime(client))

    assert len(client.calls) == 2
    assert client.calls[0]["tools"] == []
    assert client.calls[0]["think"] is True
    assert client.calls[1]["think"] is False
    assert any(
        "No tools are available or required for this turn" in str(item.get("content") or "")
        for item in client.calls[1]["messages"]
    )
    assert messages[-1]["content"] == "Hello! How can I help?"


def test_zero_tool_stray_tool_call_with_usable_prose_keeps_answer(monkeypatch):
    from al_agent import turn_engine as te

    class ContentAndInvalidCallClient:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return iter([{
                "message": {
                    "content": "Hello! How can I help you today?",
                    "tool_calls": [{"function": {"name": "ghost_tool", "arguments": {}}}],
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 10,
            }])

    client = ContentAndInvalidCallClient()
    _configure_zero_tool_turn_test(monkeypatch, te)
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(messages, "Hello?", False, runtime_overrides=_zero_tool_runtime(client))

    assert client.calls == 1
    assert messages[-1]["content"] == "Hello! How can I help you today?"


def test_repeated_zero_tool_protocol_violation_reports_specific_failure(monkeypatch):
    from al_agent import turn_engine as te

    class AlwaysInvalidClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return iter([{
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "ghost_tool", "arguments": {}}}],
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 4,
            }])

    client = AlwaysInvalidClient()
    _configure_zero_tool_turn_test(monkeypatch, te)
    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(messages, "Hello?", False, runtime_overrides=_zero_tool_runtime(client))

    assert len(client.calls) == 2
    assert client.calls[1]["think"] is False
    assert "zero-tool turn" in messages[-1]["content"]
    assert "direct-answer recovery also failed" in messages[-1]["content"]
