import json
import sqlite3
from pathlib import Path

from PIL import Image


def _configure(monkeypatch, tmp_path):
    import tools.user_profile as profile

    workspace = tmp_path / "workspace"
    memory = tmp_path / "memory"
    workspace.mkdir()
    memory.mkdir()
    monkeypatch.setattr(profile, "WORKSPACE_ROOT", workspace.resolve())
    monkeypatch.setattr(profile, "PROFILE_DIR", (memory / "profile").resolve())
    monkeypatch.setattr(profile, "PROFILE_IMAGE_PATH", (memory / "profile" / "user_picture.png").resolve())
    monkeypatch.setattr(profile, "DB_PATH", str(memory / "knowledge.db"))
    profile.init_user_profile_db()
    return profile, workspace, memory


def test_set_profile_image_normalizes_and_persists(monkeypatch, tmp_path):
    profile, workspace, _ = _configure(monkeypatch, tmp_path)
    from tools.media import PROFILE_MEDIA_REFERENCE, unpack_media_result
    source = workspace / "me.jpg"
    Image.new("RGB", (1400, 900), (200, 160, 120)).save(source, format="JPEG")

    text, images = unpack_media_result(profile.set_profile_image(str(source)))
    result = json.loads(text)

    assert result["ok"] is True
    assert result["profile_image"] == PROFILE_MEDIA_REFERENCE
    assert result["attached"] is True
    assert images == [PROFILE_MEDIA_REFERENCE]
    target = profile.PROFILE_IMAGE_PATH
    assert target.is_file()
    with Image.open(target) as saved:
        assert saved.format == "PNG"
        assert max(saved.size) <= 1024
    with sqlite3.connect(profile.DB_PATH) as conn:
        row = conn.execute("SELECT fact FROM memory WHERE topic='user_picture'").fetchone()
    assert row and str(profile.PROFILE_IMAGE_PATH) in row[0]


def test_profile_image_info_returns_attachable_logical_reference(monkeypatch, tmp_path):
    profile, workspace, _ = _configure(monkeypatch, tmp_path)
    from tools.media import PROFILE_MEDIA_REFERENCE, unpack_media_result
    source = workspace / "me.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(source)
    profile.set_profile_image(str(source))

    text, images = unpack_media_result(profile.profile_image_info())
    payload = json.loads(text)

    assert payload == {
        "present": True,
        "profile_image": PROFILE_MEDIA_REFERENCE,
        "attached": True,
    }
    assert images == [PROFILE_MEDIA_REFERENCE]
    assert "/app/memory" not in text
    state = profile.get_onboarding_state()
    assert state["profile_image"] == PROFILE_MEDIA_REFERENCE
    assert state["profile_image_present"] is True


def test_profile_image_info_without_image_is_plain_absence(monkeypatch, tmp_path):
    profile, _, _ = _configure(monkeypatch, tmp_path)
    from tools.media import unpack_media_result

    text, images = unpack_media_result(profile.profile_image_info())

    assert json.loads(text) == {"present": False, "profile_image": "", "attached": False}
    assert images == []


def test_profile_image_migrates_legacy_user_photo_from_chat_history(monkeypatch, tmp_path):
    profile, workspace, _ = _configure(monkeypatch, tmp_path)
    source = workspace / "legacy_face.png"
    Image.new("RGB", (320, 320), (20, 40, 80)).save(source)

    with sqlite3.connect(profile.DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS memory (topic TEXT PRIMARY KEY, fact TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, name TEXT, extra TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("INSERT INTO memory(topic, fact) VALUES ('user_photo', 'This is a photo of the user')")
        conn.execute("INSERT INTO chat_history(role, content) VALUES ('user', ?)", (f"describe it Attached file: {source}",))
        conn.execute("INSERT INTO chat_history(role, content) VALUES ('assistant', 'description')")
        conn.execute("INSERT INTO chat_history(role, content) VALUES ('user', 'remember that this is a photo of me')")

    migrated = profile.get_profile_image_path(migrate_legacy=True)
    assert migrated == profile.PROFILE_IMAGE_PATH
    assert migrated.is_file()


def test_profile_tool_is_selected_for_profile_picture_followup_context():
    import tools

    names = {
        schema["function"]["name"]
        for schema in tools.select_tool_schemas(
            "yes",
            context_text="Would you like to use /app/workspace/uploads/me.png as your profile picture?",
            max_tools=12,
        )
    }
    assert "set_profile_image" in names


def test_relevant_profile_context_does_not_volunteer_identity_on_unrelated_turn(tmp_path, monkeypatch):
    from tools import user_profile

    db = str(tmp_path / "profile-relevance.db")
    profile_dir = tmp_path / "profile"
    monkeypatch.setattr(user_profile, "DB_PATH", db)
    monkeypatch.setattr(user_profile, "PROFILE_DIR", profile_dir)
    monkeypatch.setattr(user_profile, "PROFILE_IMAGE_PATH", profile_dir / "user_picture.png")
    user_profile.init_user_profile_db()
    user_profile.complete_onboarding_profile(
        name="Ron", role="Developer", timezone="America/Toronto",
        location="London, Ontario, Canada", interests=["Linux", "AI"], reset=True,
    )

    greeting = user_profile.get_relevant_user_prompt_context("Good evening")
    assert "Ron" not in greeting and "London" not in greeting and "Linux" not in greeting
    unrelated = user_profile.get_relevant_user_prompt_context("Is God real?")
    assert "Ron" not in unrelated
    assert "London" not in unrelated
    assert "Linux" not in unrelated

    weather = user_profile.get_relevant_user_prompt_context("What is the weather right now?")
    assert "London, Ontario, Canada" in weather
    assert "Ron" not in weather

    explicit = user_profile.get_relevant_user_prompt_context("What do you know about me?")
    assert "Ron" in explicit
    assert "London, Ontario, Canada" in explicit
