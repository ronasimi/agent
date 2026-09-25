import json
from pathlib import Path

from al_agent.prompts import SYSTEM_POLICY
from tools import web




def test_extract_main_text_prefers_article_and_drops_site_chrome():
    html = """
    <html><body>
      <header><a>Home</a><a>Weather</a><a>Sports</a></header>
      <div class="advertisement">BUY THIS PRODUCT NOW</div>
      <main>
        <article>
          <h1>City council approves transit plan</h1>
          <p>London city council approved a new transit plan after a lengthy public meeting on Tuesday evening.</p>
          <p>The plan adds service on several major routes and is scheduled to begin next spring.</p>
          <p>Officials said implementation details will be published after the final budget review.</p>
        </article>
      </main>
      <section class="related-stories"><a>Celebrity story</a><a>Shopping guide</a></section>
      <footer>Privacy Cookies Careers Contact Us</footer>
    </body></html>
    """
    text = web.extract_main_text(html)
    assert "City council approves transit plan" in text
    assert "adds service on several major routes" in text
    assert "BUY THIS PRODUCT NOW" not in text
    assert "Celebrity story" not in text
    assert "Privacy Cookies" not in text
    assert "Home Weather Sports" not in text


def test_browse_url_returns_extracted_article_not_navigation(monkeypatch):
    html = """
    <html><body>
      <nav>HOME NEWS WEATHER SPORTS SHOP</nav>
      <article>
        <h1>Test headline</h1>
        <p>This is the first substantive paragraph of the test article and contains enough prose for extraction.</p>
        <p>This is the second substantive paragraph with additional details that belong to the article body.</p>
      </article>
      <aside>Sponsored links and recommendations</aside>
    </body></html>
    """
    monkeypatch.setattr(web, "fetch_text", lambda *a, **k: ("https://example.test/story", "text/html", html))
    result = web.browse_url("https://example.test/story")
    assert "Test headline" in result
    assert "first substantive paragraph" in result
    assert "HOME NEWS WEATHER SPORTS SHOP" not in result
    assert "Sponsored links" not in result






def test_generic_execution_timeout_schema_matches_runtime_clamp():
    from tools.system import execute_python, execute_shell
    from tools.tool_registry import function_schema

    for func in (execute_shell, execute_python):
        timeout = function_schema(func)["function"]["parameters"]["properties"]["timeout"]
        assert timeout["minimum"] == 1
        assert timeout["maximum"] == 120


def test_browse_url_reuses_active_browser_page_before_network_fetch(monkeypatch):
    from tools import web
    import tools.browser_ui as browser_ui

    monkeypatch.setattr(browser_ui, "reuse_loaded_page", lambda url, mode="text", limit=50000: {
        "source": "browser_session", "url": url, "content_type": "text/html", "text": "Dynamic SPA state"
    })
    monkeypatch.setattr(web, "fetch_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network fetch should not run")))
    result = web.browse_url("https://example.com/app")
    assert "Dynamic SPA state" in result
    assert "source=browser-session" in result


def test_page_metadata_and_links_reuse_active_browser_dom(monkeypatch):
    from tools import web_research
    import tools.browser_ui as browser_ui

    def fake_reuse(url, *, mode="text", limit=50000):
        if mode == "metadata":
            return {"url": url, "title": "Live App", "meta": {"description": "live"}, "canonical": url, "source": "browser_session"}
        if mode == "links":
            return {"source": url, "source_kind": "browser_session", "links": [
                {"url": "https://example.com/next#frag", "text": "Next"},
                {"url": "https://other.test/skip", "text": "Other"},
            ]}
        return None

    monkeypatch.setattr(browser_ui, "reuse_loaded_page", fake_reuse)
    monkeypatch.setattr(web_research, "fetch_bytes", lambda *a, **k: (_ for _ in ()).throw(AssertionError("metadata refetch should not run")))
    monkeypatch.setattr(web_research, "_page", lambda *a, **k: (_ for _ in ()).throw(AssertionError("link refetch should not run")))

    metadata = json.loads(web_research.page_metadata("https://example.com/app"))
    assert metadata["title"] == "Live App"
    assert metadata["source"] == "browser_session"

    links = json.loads(web_research.page_links("https://example.com/app", same_domain=True, limit=10))
    assert links["source_kind"] == "browser_session"
    assert links["links"] == [{"url": "https://example.com/next", "text": "Next", "same_domain": True}]
