from pathlib import Path

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
