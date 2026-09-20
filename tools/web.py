"""Safe web search and browsing helpers."""
from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .netutil import fetch_text
from .task_requirements import canonicalize_location, is_implementation_request

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
    location: str = "",
    timelimit: str = "d",
    region: str = "ca-en",
    max_results: int = 8,
) -> str:
    """Search current news and return bounded, optionally location-scoped headlines."""
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
    location = canonicalize_location(location)
    effective_query = query
    if location:
        effective_query = f"{location} local news {query}"[:1000]

    def compact_rows(items: list[dict]) -> list[dict]:
        compact: list[dict] = []
        seen_urls: set[str] = set()
        for item in items:
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
        return compact

    try:
        from ddgs import DDGS
        client = DDGS()
        results = list(client.news(
            query=effective_query, region=region, timelimit=(window or None), max_results=max(limit * 2, limit + 4)
        ))
        compact = compact_rows(results)
        if location:
            compact = _location_scoped_news_rows(compact, location, limit=limit)
            # A one-day index can be sparse for smaller cities. Retry once with a
            # broader time window and a shorter, more locality-heavy query; never
            # fall through to unrelated global headlines.
            if not compact and window in {"", "d"}:
                retry = list(client.news(
                    query=f"{location} local news",
                    region=region,
                    timelimit="w",
                    max_results=max(limit * 2, limit + 4),
                ))
                compact = _location_scoped_news_rows(compact_rows(retry), location, limit=limit)
        return json.dumps(compact[:limit], ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: news search failed: {exc}"


def _location_scoped_news_rows(rows: list[dict], location: str, *, limit: int = 8) -> list[dict]:
    """Rank and reject location-conflicting news results deterministically."""
    canonical = canonicalize_location(location)
    segments = [segment.strip() for segment in canonical.split(",") if segment.strip()]
    if not segments:
        return rows[:limit]
    city_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", segments[0]) if len(token) > 1}
    qualifier_tokens = {
        token.lower()
        for segment in segments[1:]
        for token in re.findall(r"[A-Za-z0-9]+", segment)
        if len(token) > 2
    }
    country = segments[-1].lower() if len(segments) > 1 else ""
    country_tlds = {
        "canada": ".ca", "united kingdom": ".uk", "australia": ".au", "new zealand": ".nz",
    }
    expected_tld = country_tlds.get(country, "")
    conflicting_tlds = {suffix for name, suffix in country_tlds.items() if name != country and suffix != expected_tld}
    local_source_re = re.compile(r"\b(?:news|times|free press|gazette|journal|observer|post|herald|chronicle)\b", re.I)
    ranked: list[tuple[int, int, dict]] = []
    for index, item in enumerate(rows):
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        source = str(item.get("source") or "")
        url = str(item.get("url") or "")
        haystack = " ".join((title, snippet, source, url)).lower()
        tokens = set(re.findall(r"[a-z0-9]+", haystack))
        city_match = bool(city_tokens) and city_tokens.issubset(tokens)
        if not city_match:
            continue
        score = 5
        score += 3 * len(qualifier_tokens & tokens)
        host = (urlparse(url).hostname or "").lower()
        if expected_tld and host.endswith(expected_tld):
            score += 2
        if any(host.endswith(suffix) for suffix in conflicting_tlds):
            score -= 8
        if city_tokens.issubset(set(re.findall(r"[a-z0-9]+", source.lower()))) and local_source_re.search(source):
            score += 2
        # A qualified location (for example London, Ontario, Canada) needs one
        # signal beyond the ambiguous city name: province/country, matching TLD,
        # or a city-branded local publication.
        threshold = 6 if qualifier_tokens else 5
        if score >= threshold:
            ranked.append((score, -index, item))
    ranked.sort(key=lambda row: (-row[0], -row[1]))
    return [item for _score, _index, item in ranked[:max(1, min(int(limit), 12))]]


def wiki_search(query: str = "") -> str:
    """Search Wikipedia for concise encyclopedic background information."""
    query = str(query).strip()
    if not query:
        return "Error: Missing required 'query' parameter."
    if len(query) > 1000:
        return "Error: Query is limited to 1000 characters."
    try:
        import wikipedia
        page = wikipedia.page(query, auto_suggest=True, preload=False)
        summary = wikipedia.summary(page.title, sentences=4, auto_suggest=False)
        return json.dumps(
            {
                "title": str(page.title or query)[:300],
                "url": str(page.url or "")[:1200],
                "summary": str(summary or "")[:5000],
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return f"Error: Wikipedia search failed: {exc}"


def format_encyclopedia_result(content: str) -> str:
    """Render only the structured fields returned by :func:`wiki_search`."""
    try:
        payload = json.loads(str(content or ""))
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    title = str(payload.get("title") or "").strip()
    summary = str(payload.get("summary") or "").strip()
    url = str(payload.get("url") or "").strip()
    if not summary:
        return ""
    rendered = summary
    if title and url.startswith(("http://", "https://")):
        rendered += f"\n\nSource: Wikipedia — [{title}]({url})"
    elif url.startswith(("http://", "https://")):
        rendered += f"\n\nSource: Wikipedia — {url}"
    return rendered


def is_simple_encyclopedic_request(user_request: str) -> bool:
    """Return whether a request only asks for a short definition/identity."""
    from .grounding import encyclopedic_lookup_query

    subject = encyclopedic_lookup_query(user_request)
    if not subject:
        return False
    text = " ".join(str(user_request or "").lower().split())
    return not re.search(
        r"\b(?:compare|analy[sz]e|critic|debate|argument|history of|timeline|examples?|"
        r"write|essay|report|deep dive|detailed|comprehensive|why|how)\b",
        text,
    )


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
    if not re.search(r"\b(?:news|headlines?)\b", text) or is_implementation_request(user_request):
        return False
    return not re.search(r"\b(?:why|explain|analy[sz]e|compare|impact|opinion|summari[sz]e .*story|details? about)\b", text)


def format_news_results(content: str, *, limit: int = 6, location: str = "") -> str:
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
    heading = f"**Latest local headlines for {canonicalize_location(location)}**" if str(location or "").strip() else "**Latest headlines**"
    lines = [heading, ""]
    for item in rows:
        title = str(item.get("title") or "").strip()
        source = str(item.get("source") or "").strip()
        date = str(item.get("date") or "").strip()
        url = str(item.get("url") or "").strip()
        meta = " · ".join(x for x in (source, date) if x)
        lines.append(f"- [{title}]({url})" + (f" — {meta}" if meta else ""))
    return "\n".join(lines)
