from pathlib import Path


def test_skill_metadata_is_relevant_and_full_body_is_lazy(tmp_path, monkeypatch):
    from tools import skills

    monkeypatch.setattr(skills, "SKILLS_DIR", tmp_path)
    (tmp_path / "linux-tuning.md").write_text(
        "---\nname: Linux Tuning\ndescription: Tune Linux latency and memory behavior\ntags: [linux, sysctl, latency]\n---\n"
        "# Linux Tuning\nSECRET_FULL_SKILL_BODY\nUse bounded measurements before changes.\n",
        encoding="utf-8",
    )
    index = skills.render_relevant_skill_index("reduce linux latency", limit=3)
    assert "linux-tuning" in index
    assert "Tune Linux latency" in index
    assert "SECRET_FULL_SKILL_BODY" not in index
    loaded = skills.load_skill("linux-tuning")
    payload = __import__("json").loads(loaded)
    assert "SECRET_FULL_SKILL_BODY" in payload["content"]
    assert payload["has_more"] is False


def test_skill_search_does_not_match_unrelated_query(tmp_path, monkeypatch):
    from tools import skills

    monkeypatch.setattr(skills, "SKILLS_DIR", tmp_path)
    (tmp_path / "git.md").write_text("# Git Repair\nRepair git branches safely.", encoding="utf-8")
    assert skills.relevant_skill_metadata("weather forecast", limit=3) == []
