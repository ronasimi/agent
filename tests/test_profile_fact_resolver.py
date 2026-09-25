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
