import base64
import json
from pathlib import Path

from PIL import Image
from tools import media


def test_media_result_round_trip():
    result = media.media_result("page text", ["/app/workspace/a.png", "/app/workspace/b.jpg"])
    text, images = media.unpack_media_result(result)
    assert text == "page text"
    assert images == ["/app/workspace/a.png", "/app/workspace/b.jpg"]


def test_plain_tool_result_has_no_media():
    text, images = media.unpack_media_result("normal tool output")
    assert text == "normal tool output"
    assert images == []


def test_attach_media_accepts_workspace_image(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "WORKSPACE_ROOT", tmp_path)
    image = tmp_path / "screen.png"
    image.write_bytes(b"fake")
    result = media.attach_media("screen.png", "Inspect the alert banner")
    text, images = media.unpack_media_result(result)
    assert "Inspect the alert banner" in text
    assert images == [str(image.resolve())]


def test_attach_media_rejects_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "WORKSPACE_ROOT", tmp_path)
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"fake")
    result = media.attach_media(str(outside))
    assert isinstance(result, str)
    assert result.startswith("Error:")


def test_attach_media_accepts_only_the_canonical_profile_image(tmp_path, monkeypatch):
    from tools import user_profile

    profile = tmp_path / "memory" / "profile" / "user_picture.png"
    profile.parent.mkdir(parents=True)
    profile.write_bytes(b"fake")
    monkeypatch.setattr(user_profile, "PROFILE_IMAGE_PATH", profile)

    for reference in (media.PROFILE_MEDIA_REFERENCE, str(profile)):
        result = media.attach_media(reference, "Inspect the profile picture")
        text, images = media.unpack_media_result(result)
        assert "Inspect the profile picture" in text
        assert images == [media.PROFILE_MEDIA_REFERENCE]

    unrelated = profile.parent / "private.png"
    unrelated.write_bytes(b"fake")
    rejected = media.attach_media(str(unrelated))
    assert isinstance(rejected, str)
    assert rejected.startswith("Error:")


def test_profile_media_reference_can_be_encoded_and_inspected(tmp_path, monkeypatch):
    from al_agent.prompts import encode_image
    from tools import user_profile
    from tools.primitive_modules.media import image_info

    profile = tmp_path / "profile" / "user_picture.png"
    profile.parent.mkdir(parents=True)
    Image.new("RGB", (48, 32), (20, 40, 60)).save(profile)
    monkeypatch.setattr(user_profile, "PROFILE_IMAGE_PATH", profile)

    encoded = encode_image(media.PROFILE_MEDIA_REFERENCE)
    assert encoded is not None
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")
    info = json.loads(image_info(media.PROFILE_MEDIA_REFERENCE))
    assert info["width"] == 48
    assert info["height"] == 32
    assert info["format"] == "PNG"


def test_image_transformers_accept_profile_reference_and_self_attach(tmp_path, monkeypatch):
    from tools import user_profile, workspace as workspace_tools
    from tools.primitive_modules.media import image_convert, image_crop, image_resize

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile = tmp_path / "profile" / "user_picture.png"
    profile.parent.mkdir(parents=True)
    Image.new("RGB", (80, 60), (20, 40, 60)).save(profile)
    monkeypatch.setattr(user_profile, "PROFILE_IMAGE_PATH", profile)
    monkeypatch.setattr(workspace_tools, "WORKSPACE_DIR", str(workspace.resolve()))

    cases = [
        (image_resize(media.PROFILE_MEDIA_REFERENCE, "resized.png", 40, 40), "resized.png"),
        (image_crop(media.PROFILE_MEDIA_REFERENCE, "cropped.png", 0, 0, 20, 20), "cropped.png"),
        (image_convert(media.PROFILE_MEDIA_REFERENCE, "converted.webp", "WEBP"), "converted.webp"),
    ]
    for result, filename in cases:
        text, images = media.unpack_media_result(result)
        payload = json.loads(text)
        expected = str((workspace / filename).resolve())
        assert payload["path"] == expected
        assert images == [expected]
        assert Path(expected).is_file()


def test_rendered_document_page_self_attaches(tmp_path, monkeypatch):
    from tools import workspace as workspace_tools
    from tools.primitive_modules import documents

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "document.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(workspace_tools, "WORKSPACE_DIR", str(workspace.resolve()))
    monkeypatch.setattr(documents.shutil, "which", lambda name: "/usr/bin/pdftoppm")

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(argv, **kwargs):
        Path(argv[-1] + ".png").write_bytes(b"fake-png")
        return Result()

    monkeypatch.setattr(documents.subprocess, "run", fake_run)
    result = documents.render_document_page("document.pdf", page=1)
    text, images = media.unpack_media_result(result)
    payload = json.loads(text)
    expected = str((workspace / "document-page-1.png").resolve())
    assert payload["path"] == expected
    assert payload["created"] is True
    assert images == [expected]
