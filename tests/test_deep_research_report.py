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


def test_claim_ledger_rejects_non_verbatim_support_and_keeps_source_scope():
    from tools.research_factuality import validate_ledger_batch

    source_map = {
        "S1": {
            "source_id": "S1",
            "title": "Primary source",
            "url": "https://example.com/source",
            "retrieved_at": "2026-09-20T00:00:00+00:00",
            "raw_content": "Cloudflare announced a reduction of more than 1,100 employees in May 2026.",
        }
    }
    payload = {
        "sources": [{
            "source_id": "S1",
            "claims": [
                {
                    "claim": "Cloudflare announced a reduction of more than 1,100 employees in May 2026.",
                    "support_excerpt": "Cloudflare announced a reduction of more than 1,100 employees in May 2026.",
                    "support_type": "organization_statement",
                    "confidence": "high",
                    "qualification": "",
                },
                {
                    "claim": "The layoffs funded GPU clusters.",
                    "support_excerpt": "The layoffs funded GPU clusters.",
                    "support_type": "direct_fact",
                    "confidence": "high",
                    "qualification": "",
                },
            ],
            "limitations": [],
        }]
    }
    ledger = validate_ledger_batch(payload, source_map)
    assert len(ledger) == 1
    assert [claim["claim"] for claim in ledger[0]["claims"]] == [
        "Cloudflare announced a reduction of more than 1,100 employees in May 2026."
    ]


def test_claim_ledger_rendering_preserves_support_type_and_exact_excerpt():
    from tools.research_factuality import render_claim_ledger

    ledger = [{
        "source_id": "S4",
        "title": "Analysis",
        "url": "https://example.com/analysis",
        "authority_hint": "web_source",
        "claims": [{
            "claim": "Analysts questioned the pace of the transition.",
            "support_excerpt": "analysts questioned the pace of the transition",
            "support_type": "analysis_opinion",
            "confidence": "medium",
            "qualification": "attributed analysis",
        }],
        "limitations": ["Opinion source"],
    }]
    rendered = render_claim_ledger(ledger, ["S4"])
    assert "analysis_opinion" in rendered
    assert "Exact support:" in rendered
    assert "Opinion source" in rendered


def test_safe_ledger_fallback_contains_only_selected_source_claims():
    from tools.research_factuality import safe_ledger_fallback

    ledger = [
        {"source_id": "S1", "claims": [{"claim": "Supported one.", "qualification": ""}]},
        {"source_id": "S2", "claims": [{"claim": "Supported two.", "qualification": ""}]},
    ]
    text = safe_ledger_fallback(ledger, ["S2"])
    assert "Supported two" in text
    assert "[S2]" in text
    assert "Supported one" not in text
