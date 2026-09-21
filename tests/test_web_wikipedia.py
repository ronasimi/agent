from __future__ import annotations

import json


def test_wiki_search_uses_bounded_mediawiki_json_api(monkeypatch):
    from tools import web

    body = json.dumps({
        "query": {
            "pages": {
                "1": {
                    "pageid": 1,
                    "index": 1,
                    "title": "Linux",
                    "fullurl": "https://en.wikipedia.org/wiki/Linux",
                    "extract": "Linux is an operating system kernel. It is open source.",
                }
            }
        }
    })
    seen = {}

    def fake_fetch(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        return url, "application/json", body

    monkeypatch.setattr(web, "fetch_text", fake_fetch)
    payload = json.loads(web.wiki_search("Linux"))
    assert payload["title"] == "Linux"
    assert payload["url"].endswith("/wiki/Linux")
    assert "generator=search" in seen["url"]
    assert seen["kwargs"]["max_bytes"] == 512 * 1024


def test_wiki_search_reports_non_json_provider_response(monkeypatch):
    from tools import web

    monkeypatch.setattr(
        web,
        "fetch_text",
        lambda url, **kwargs: (url, "text/plain", "upstream maintenance"),
    )
    result = web.wiki_search("Linux")
    assert result.startswith("Error: Wikipedia search failed:")
    assert "non-JSON" in result
