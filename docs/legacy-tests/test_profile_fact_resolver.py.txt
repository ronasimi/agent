from __future__ import annotations


def _configure_profile(tmp_path, monkeypatch):
    from tools import user_profile

    db = str(tmp_path / "profile-facts.db")
    profile_dir = tmp_path / "profile"
    monkeypatch.setattr(user_profile, "DB_PATH", db)
    monkeypatch.setattr(user_profile, "PROFILE_DIR", profile_dir)
    monkeypatch.setattr(user_profile, "PROFILE_IMAGE_PATH", profile_dir / "user_picture.png")
    user_profile.init_user_profile_db()
    user_profile.complete_onboarding_profile(
        name="Ron Asimi",
        role="Developer",
        timezone="America/Toronto",
        location="London, Ontario, Canada",
        email="ron@example.invalid",
        interests=["Linux", "AI"],
        response_style="detailed",
        research_depth="deep",
        reset=True,
    )
    return user_profile


def test_profile_fact_resolver_reads_oobe_fields_without_model(tmp_path, monkeypatch):
    profile = _configure_profile(tmp_path, monkeypatch)

    name = profile.resolve_profile_fact_query("what is my name?")
    assert name["matched"] is True and name["resolved"] is True
    assert name["facts"] == {"name": "Ron Asimi"}
    assert name["response"] == "Your name is Ron Asimi."

    location = profile.resolve_profile_fact_query("what is my saved location?")
    assert location["facts"] == {"location": "London, Ontario, Canada"}

    timezone = profile.resolve_profile_fact_query("what is my timezone?")
    assert timezone["facts"] == {"timezone": "America/Toronto"}

    role = profile.resolve_profile_fact_query("what is my role?")
    assert role["facts"] == {"role": "Developer"}

    interests = profile.resolve_profile_fact_query("what are my interests?")
    assert interests["facts"] == {"interests": ["Linux", "AI"]}

    style = profile.resolve_profile_fact_query("what is my response style?")
    assert style["facts"] == {"response_style": "detailed"}

    depth = profile.resolve_profile_fact_query("what is my research depth?")
    assert depth["facts"] == {"research_depth": "deep"}


def test_profile_fact_resolver_does_not_intercept_mutation_or_missing_fact(tmp_path, monkeypatch):
    profile = _configure_profile(tmp_path, monkeypatch)
    assert profile.resolve_profile_fact_query("change my name to Alice")["matched"] is False

    profile.set_user_identity(name="Ron", role="Developer", timezone="America/Toronto", email="", interests=[])
    missing = profile.resolve_profile_fact_query("what is my email?")
    assert missing["matched"] is True
    assert missing["resolved"] is False
    assert missing["missing"] == ["email"]


def test_turn_engine_short_circuits_resolved_profile_fact_before_ollama(monkeypatch):
    from al_agent import turn_engine as te

    class NeverCalledClient:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            raise AssertionError("Ollama must not be called for a resolved profile fact")

    client = NeverCalledClient()
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "resolve_profile_fact_query", lambda text: {
        "matched": True,
        "resolved": True,
        "requested": ["name"],
        "facts": {"name": "Ron Asimi"},
        "missing": [],
        "response": "Your name is Ron Asimi.",
    })
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        "what is my name?",
        False,
        runtime_overrides={
            "OLLAMA": client,
            "record_monitor_state": lambda *a, **k: None,
            "append_and_save": lambda rows, item: rows.append(item),
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: (_ for _ in ()).throw(AssertionError("inference lock should not be acquired")),
            "release_inference_lock": lambda _lock: None,
            "queue_compaction_if_needed": lambda *a, **k: None,
        },
    )

    assert client.calls == 0
    assert messages[-1]["content"] == "Your name is Ron Asimi."


def test_turn_engine_falls_back_to_model_when_profile_fact_is_missing(monkeypatch):
    from al_agent import turn_engine as te

    class OneShotClient:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return iter([{
                "message": {"content": "I don't have a saved email address for you."},
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 8,
            }])

    client = OneShotClient()
    monkeypatch.setattr(te, "WORKING_STATE_ENABLED", False)
    monkeypatch.setattr(te, "RECIPES_ENABLED", False)
    monkeypatch.setattr(te, "LOOP_VALIDATOR_ENABLED", False)
    monkeypatch.setattr(te, "MODEL_TRACE_ENABLED", False)
    monkeypatch.setattr(te, "resolve_profile_fact_query", lambda text: {
        "matched": True, "resolved": False, "requested": ["email"],
        "facts": {}, "missing": ["email"], "response": "",
    })
    monkeypatch.setattr(te, "get_conversation_summary", lambda: "")
    monkeypatch.setattr(te, "build_memory_context", lambda *_: "")
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda *_: "")
    monkeypatch.setattr(te, "get_user_location", lambda: "")
    monkeypatch.setattr(te, "evict_report_model_for_interactive", lambda: None)
    monkeypatch.setattr(te, "_prune_compacted_history", lambda _messages: None)
    monkeypatch.setattr(te, "log_perf_stats", lambda *a, **k: None)

    messages = [{"role": "system", "content": "system"}]
    te.handle_user_turn(
        messages,
        "what is my email?",
        False,
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

    assert client.calls == 1
    assert messages[-1]["content"] == "I don't have a saved email address for you."
