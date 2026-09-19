"""Safe web search and browsing helpers."""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .netutil import fetch_text

_TRACKING_SEARCH_HOSTS = {"googleadservices.com", "www.googleadservices.com"}

def _search_result_url_allowed(value: str) -> bool:
    """Drop obvious ad/tracking redirect results before they reach the model."""
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if host in _TRACKING_SEARCH_HOSTS:
        return False
    if host.endswith("bing.com") and path.startswith("/aclick"):
        return False
    if any(token in path for token in ("/aclk", "/pagead/aclk")):
        return False
    return True


def web_search(query: str = "") -> str:
    """Search the public web for current information and return a small JSON result set."""
    query = str(query).strip()
    if not query:
        return "Error: Missing required 'query' parameter."
    if len(query) > 1000:
        return "Error: Query is limited to 1000 characters."
    try:
        from ddgs import DDGS
        # Ask for a few extra candidates so removing sponsored redirect links
        # still leaves a useful bounded result set.
        results = list(DDGS().text(query, max_results=8))
        compact = []
        seen_urls: set[str] = set()
        for item in results:
            url = str(item.get("href", "") or "").strip()
            if not _search_result_url_allowed(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            compact.append({
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("body", ""),
            })
            if len(compact) >= 5:
                break
        return json.dumps(compact, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: web search failed: {exc}"


def news_search(
    query: str = "",
    timelimit: str = "d",
    region: str = "ca-en",
    max_results: int = 8,
) -> str:
    """Search current news and return bounded structured headline metadata."""
    query = str(query).strip()
    if not query:
        return "Error: Missing required 'query' parameter."
    if len(query) > 1000:
        return "Error: Query is limited to 1000 characters."
    window = str(timelimit or "").strip().lower()
    if window not in {"", "d", "w", "m"}:
        return "Error: timelimit must be one of '', 'd', 'w', or 'm'."
    try:
        limit = max(1, min(int(max_results), 12))
    except (TypeError, ValueError):
        limit = 8
    region = str(region or "ca-en").strip()[:24] or "ca-en"
    try:
        from ddgs import DDGS
        results = list(DDGS().news(
            query=query, region=region, timelimit=(window or None), max_results=max(limit + 3, limit)
        ))
        compact = []
        seen_urls: set[str] = set()
        for item in results:
            url = str(item.get("url", "") or "").strip()
            if not _search_result_url_allowed(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            compact.append({
                "date": str(item.get("date", "") or "")[:80],
                "title": str(item.get("title", "") or "")[:500],
                "url": url,
                "snippet": str(item.get("body", "") or "")[:1200],
                "source": str(item.get("source", "") or "")[:200],
            })
            if len(compact) >= limit:
                break
        return json.dumps(compact, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: news search failed: {exc}"


def wiki_search(query: str = "") -> str:
    """Search Wikipedia for encyclopedic background information."""
    query = str(query).strip()
    if not query:
        return "Error: Missing required 'query' parameter."
    if len(query) > 1000:
        return "Error: Query is limited to 1000 characters."
    try:
        import wikipedia
        return wikipedia.summary(query, sentences=4, auto_suggest=True)
    except Exception as exc:
        return f"Error: Wikipedia search failed: {exc}"


def browse_url(url: str = "") -> str:
    """Fetch a public HTTP(S) URL with redirect, size, and private-network protections and extract readable text."""
    if not str(url).strip():
        return "Error: Missing required 'url' parameter."
    try:
        final_url, content_type, body = fetch_text(url, max_bytes=2 * 1024 * 1024)
        if content_type in {"application/json", "application/xml", "text/xml", "text/plain"}:
            text = body
        else:
            soup = BeautifulSoup(body, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
                element.decompose()
            text = "\n".join(soup.stripped_strings)
        text = text[:20000]
        return f"URL: {final_url}\nContent-Type: {content_type}\n\n{text or 'The page returned no readable text content.'}"
    except Exception as exc:
        return f"Error: browsing URL failed: {exc}"


def is_simple_headline_request(user_request: str) -> bool:
    """Return whether a request only asks to list current headlines."""
    text = " ".join(str(user_request or "").lower().split())
    if not re.search(r"\b(?:news|headlines?)\b", text):
        return False
    return not re.search(r"\b(?:why|explain|analy[sz]e|compare|impact|opinion|summari[sz]e .*story|details? about)\b", text)


def format_news_results(content: str, *, limit: int = 6) -> str:
    """Render structured news-search output without relying on model tool-call compliance."""
    try:
        payload = json.loads(str(content or ""))
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, list):
        return ""
    rows = [item for item in payload if isinstance(item, dict) and item.get("title") and item.get("url")][:max(1, min(int(limit), 10))]
    if not rows:
        return ""
    lines = ["**Latest headlines**", ""]
    for item in rows:
        title = str(item.get("title") or "").strip()
        source = str(item.get("source") or "").strip()
        date = str(item.get("date") or "").strip()
        url = str(item.get("url") or "").strip()
        meta = " · ".join(x for x in (source, date) if x)
        lines.append(f"- [{title}]({url})" + (f" — {meta}" if meta else ""))
    return "\n".join(lines)
