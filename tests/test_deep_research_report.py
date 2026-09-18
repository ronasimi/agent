from pathlib import Path

from tools.deep_research import _extract_image_candidates
from tools import pdf_generator


def test_image_candidates_prefer_social_and_large_content_images():
    html = """
    <html><head>
      <meta property="og:image" content="/hero.jpg">
    </head><body>
      <img src="/logo.png" alt="Site logo" width="80" height="80">
      <img src="/chart.png" alt="Benchmark throughput chart" width="1200" height="700">
    </body></html>
    """
    candidates = _extract_image_candidates(html, "https://example.com/article")
    urls = [item["url"] for item in candidates]
    assert urls[0] == "https://example.com/hero.jpg"
    assert "https://example.com/chart.png" in urls
    assert "https://example.com/logo.png" not in urls


def test_pdf_output_path_accepts_absolute_path_inside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(pdf_generator, "WORKSPACE", workspace.resolve())
    requested = workspace / "research" / "report.pdf"
    resolved = pdf_generator._safe_output_path(str(requested))
    assert resolved == requested.resolve()
    assert resolved.parent.is_dir()


def test_pdf_output_path_rejects_escape(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(pdf_generator, "WORKSPACE", workspace.resolve())
    outside = tmp_path / "outside.pdf"
    try:
        pdf_generator._safe_output_path(str(outside))
    except ValueError as exc:
        assert "inside /app/workspace" in str(exc)
    else:
        raise AssertionError("path escape should have been rejected")
