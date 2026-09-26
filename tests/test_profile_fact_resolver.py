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


def test_get_user_profile_returns_bounded_read_only_snapshot(tmp_path, monkeypatch):
    import json

    profile = _configure_profile(tmp_path, monkeypatch)
    payload = json.loads(profile.get_user_profile())
    assert payload["configured"] is True
    assert payload["identity"]["name"] == "Ron Asimi"
    assert payload["location"] == "London, Ontario, Canada"
    assert payload["preferences"]["response"]["style"] == "detailed"


def test_turn_engine_resolves_profile_before_model_and_records_evidence(monkeypatch):
    from al_agent import turn_engine as te

    class FakeTape:
        committed = []
        def __init__(self, *args, **kwargs):
            pass
        def render_prompt_context(self, **kwargs):
            return ""
        def recent(self, *args, **kwargs):
            return []
        def commit_turn(self, **kwargs):
            self.committed.append(kwargs)
        @staticmethod
        def collapse_tool_result(**kwargs):
            from tools.state_tape import CompactToolOutcome
            return CompactToolOutcome(
                kwargs["tool_name"], True, "User profile lookup succeeded: name=Ron Asimi.",
                kwargs.get("observation_id", ""),
            )

    class FakeWorkingState:
        begin_kwargs = None
        record_kwargs = None
        def __init__(self, *args, **kwargs):
            pass
        def begin_turn(self, **kwargs):
            type(self).begin_kwargs = kwargs
            return {}
        def record_tool_result(self, **kwargs):
            type(self).record_kwargs = kwargs
        def complete_turn(self, *, blocked=False):
            pass
        def load(self):
            return {"status": "complete"}

    monkeypatch.setattr(te, "StateTapeStore", FakeTape)
    monkeypatch.setattr(te, "WorkingStateStore", FakeWorkingState)
    monkeypatch.setattr(te, "get_active_conversation_id", lambda: "profile-turn")
    monkeypatch.setattr(te, "get_conversation_summary", lambda *_: "")
    monkeypatch.setattr(te, "resolve_profile_fact_query", lambda _text: {
        "matched": True,
        "resolved": True,
        "requested": ["name"],
        "facts": {"name": "Ron Asimi"},
        "missing": [],
        "response": "Your name is Ron Asimi.",
    })
    monkeypatch.setattr(te, "get_user_profile", lambda: '{"configured":true,"identity":{"name":"Ron Asimi"}}')
    monkeypatch.setattr(te, "get_relevant_user_prompt_context", lambda _text: "**Name**: Ron Asimi")
    monkeypatch.setattr(te, "store_tool_observation", lambda *a, **k: "profile-observation")
    monkeypatch.setattr(te, "record_monitor_state", lambda *a, **k: None)
    monkeypatch.setattr(te, "set_foreground_turn", lambda *a, **k: None)
    monkeypatch.setattr(te, "emit_event", lambda *a, **k: None)
    monkeypatch.setattr(te, "cancel_requested", lambda: False)
    monkeypatch.setattr(te, "run_loop", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model must not run")))

    next_id = iter([11, 12])
    def append(rows, item):
        item = dict(item)
        item["_db_id"] = next(next_id)
        rows.append(item)

    messages = []
    te.handle_user_turn(
        messages,
        "What is my name?",
        False,
        runtime_overrides={
            "append_and_save": append,
            "acquire_turn_lock": lambda: object(),
            "release_turn_lock": lambda _lock: None,
            "acquire_inference_lock": lambda: (_ for _ in ()).throw(AssertionError("inference lock must not be acquired")),
            "release_inference_lock": lambda _lock: None,
        },
    )

    assert any(row.get("role") == "assistant" and row.get("content") == "Your name is Ron Asimi." for row in messages)
    req = FakeWorkingState.begin_kwargs["requirements"][0]
    assert req["tool"] == "get_user_profile" and req["status"] == "satisfied"
    assert req["evidence"][0]["evidence_ref"] == "profile-observation"
    assert FakeWorkingState.record_kwargs["tool_name"] == "get_user_profile"


def test_session_prompt_includes_only_relevant_profile_context():
    from al_agent.turn_engine import build_session_system_prompt

    prompt = build_session_system_prompt(
        state_tape_context="Turn 1: prior context",
        relevant_user_context="**Name**: Ron Asimi\n**Timezone**: America/Toronto",
    )
    assert "Turn 1: prior context" in prompt
    assert "Relevant User Profile" in prompt
    assert "Ron Asimi" in prompt
