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
_NO_NEWS_RESULTS_RE = re.compile(r"\bno results found\b", re.I)
_LOCATION_COUNTRY_TERMS = {
    "canada": {"canada"},
    "united kingdom": {"united kingdom", "uk", "england", "scotland", "wales", "northern ireland"},
    "australia": {"australia"},
    "new zealand": {"new zealand"},
}


def _ddgs_news_rows(client, **kwargs) -> list[dict]:
    """Return DDGS news rows, treating the provider's empty-result exception as [].

    DDGS may raise ``No results found`` instead of returning an empty iterable.
    That is a successful zero-row retrieval, not malformed tool output. Other
    provider/transport failures still propagate to the normal error path.
    """
    try:
        return list(client.news(**kwargs))
    except Exception as exc:
        if _NO_NEWS_RESULTS_RE.search(str(exc or "")):
            return []
        raise


def _news_effective_query(query: str, location: str) -> str:
    """Add missing locality terms exactly once instead of duplicating them."""
    base = " ".join(str(query or "").split())
    canonical = canonicalize_location(location)
    if canonical:
        qtokens = set(re.findall(r"[a-z0-9]+", base.lower()))
        location_tokens = {
            token for token in re.findall(r"[a-z0-9]+", canonical.lower())
            if len(token) > 1
        }
        if location_tokens and not location_tokens.issubset(qtokens):
            base = f"{canonical} {base}".strip()
    if not re.search(r"\b(?:news|headlines?|stories?)\b", base, re.I):
        base = f"{base} news".strip()
    return base[:1000]


def _news_retry_query(location: str) -> str:
    """Build one locality-heavy fallback query for a sparse daily news index."""
    canonical = canonicalize_location(location)
    segments = [segment.strip() for segment in canonical.split(",") if segment.strip()]
    if not segments:
        return "local news"
    quoted = " ".join(f'"{segment}"' for segment in segments[:3])
    return f"{quoted} local news"[:1000]


def _has_conflicting_city_qualifier(haystack: str, city: str, country: str, qualifiers: set[str]) -> bool:
    """Reject an ambiguous city explicitly tied to another country/region."""
    city_phrase = r"\s+".join(re.escape(token) for token in re.findall(r"[a-z0-9]+", city.lower()))
    if not city_phrase or not country:
        return False
    requested_after_city = {
        token for token in qualifiers | _LOCATION_COUNTRY_TERMS.get(country, {country})
        if token
    }
    conflicting = set().union(*(terms for name, terms in _LOCATION_COUNTRY_TERMS.items() if name != country))
    normalized = re.sub(r"[^a-z0-9]+", " ", haystack.lower()).strip()
    if not normalized:
        return False
    requested_pattern = re.compile(
        rf"\b{city_phrase}\b(?:\s+\w+){{0,3}}\s+(?:"
        + "|".join(re.escape(term) for term in sorted(requested_after_city, key=len, reverse=True))
        + r")\b",
        re.I,
    ) if requested_after_city else None
    conflict_pattern = re.compile(
        rf"\b{city_phrase}\b(?:\s+\w+){{0,3}}\s+(?:"
        + "|".join(re.escape(term) for term in sorted(conflicting, key=len, reverse=True))
        + r")\b",
        re.I,
    ) if conflicting else None
    return bool(conflict_pattern and conflict_pattern.search(normalized) and not (requested_pattern and requested_pattern.search(normalized)))

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
    effective_query = _news_effective_query(query, location)

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
        results = _ddgs_news_rows(
            client, query=effective_query, region=region, timelimit=(window or None),
            max_results=max(limit * 2, limit + 4),
        )
        compact = compact_rows(results)
        if location:
            compact = _location_scoped_news_rows(compact, location, limit=limit)
            # A one-day index can be sparse for smaller cities. Retry once with a
            # broader time window and a shorter, more locality-heavy query; never
            # fall through to unrelated global headlines.
            if not compact and window in {"", "d"}:
                retry = _ddgs_news_rows(
                    client, query=_news_retry_query(location), region=region, timelimit="w",
                    max_results=max(limit * 2, limit + 4),
                )
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
        city_name = segments[0] if segments else ""
        if _has_conflicting_city_qualifier(haystack, city_name, country, qualifier_tokens):
            continue
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


def news_search_is_empty(content: str) -> bool:
    """Return True only for a valid structured zero-row news result."""
    try:
        payload = json.loads(str(content or ""))
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, list) and not payload


def format_news_no_results(*, location: str = "") -> str:
    """Render a bounded retrieval miss without claiming that no news exists."""
    canonical = canonicalize_location(location)
    where = f" for **{canonical}**" if canonical else ""
    return (
        f"I couldn't find any qualifying current local headlines{where} in the news provider's bounded search. "
        "I won't substitute unrelated or wrong-location stories."
    )


def format_news_provider_error(content: str, *, location: str = "") -> str:
    """Render one terminal provider failure for a news-only fact request."""
    canonical = canonicalize_location(location)
    where = f" for **{canonical}**" if canonical else ""
    detail = str(content or "").strip()
    detail = re.sub(r"^Error:\s*", "", detail, flags=re.I)[:240]
    suffix = f" ({detail})" if detail else ""
    return f"I couldn't retrieve current local headlines{where} because the news provider failed{suffix}."


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
