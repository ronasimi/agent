"""Safe web search and browsing helpers."""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode, urlparse
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

from .netutil import fetch_text
from .extraction import targeted_extract
from .task_requirements import canonicalize_location, is_implementation_request

_TRACKING_SEARCH_HOSTS = {"googleadservices.com", "www.googleadservices.com"}
_NO_NEWS_RESULTS_RE = re.compile(r"\bno results found\b", re.I)
_LOCATION_COUNTRY_TERMS = {
    "canada": {"canada"},
    "united kingdom": {"united kingdom", "uk", "england", "scotland", "wales", "northern ireland"},
    "australia": {"australia"},
    "new zealand": {"new zealand"},
}


_GOOGLE_NEWS_LOCALES = {
    "ca-en": ("en-CA", "CA", "CA:en"),
    "us-en": ("en-US", "US", "US:en"),
    "uk-en": ("en-GB", "GB", "GB:en"),
    "au-en": ("en-AU", "AU", "AU:en"),
}
_GENERIC_NEWS_QUERY_TOKENS = {
    "latest", "recent", "current", "today", "todays", "top", "news", "headline",
    "headlines", "story", "stories",
}


def _is_generic_news_query(query: str) -> bool:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", str(query or ""))}
    return bool(tokens) and tokens.issubset(_GENERIC_NEWS_QUERY_TOKENS)


def _google_news_rss_rows(
    query: str, *, region: str = "ca-en", timelimit: str = "d", max_results: int = 8,
) -> list[dict]:
    """Fetch a bounded Google News RSS fallback independent of DDGS.

    The fallback exists to keep ``news_search`` useful when DDGS's news backend
    is sparse, rate-limited, or temporarily unavailable.  Generic headline
    requests use the regional top-stories feed; topical/local requests use the
    RSS search endpoint with a bounded ``when:`` qualifier.
    """
    locale, gl, ceid = _GOOGLE_NEWS_LOCALES.get(str(region or "").lower(), _GOOGLE_NEWS_LOCALES["ca-en"])
    params = {"hl": locale, "gl": gl, "ceid": ceid}
    if _is_generic_news_query(query):
        url = "https://news.google.com/rss?" + urlencode(params)
    else:
        suffix = {"d": "1d", "w": "7d", "m": "30d"}.get(str(timelimit or "").lower(), "")
        rss_query = " ".join(str(query or "").split())
        if suffix and not re.search(r"\bwhen:\S+", rss_query, re.I):
            rss_query = f"{rss_query} when:{suffix}".strip()
        url = "https://news.google.com/rss/search?" + urlencode({"q": rss_query, **params})
    _final_url, _content_type, body = fetch_text(
        url, timeout=6.0, max_bytes=1024 * 1024,
        allowed_types={"application/xml", "text/xml", "application/rss+xml", "text/plain"},
    )
    root = ET.fromstring(body)
    rows: list[dict] = []
    seen_urls: set[str] = set()
    for item in root.findall(".//item"):
        title = " ".join(str(item.findtext("title") or "").split())
        link = str(item.findtext("link") or "").strip()
        if not title or not _search_result_url_allowed(link) or link in seen_urls:
            continue
        seen_urls.add(link)
        source_node = item.find("source")
        source = " ".join(str(source_node.text if source_node is not None else "").split())
        description = str(item.findtext("description") or "")
        snippet = BeautifulSoup(description, "html.parser").get_text(" ", strip=True)[:1200]
        rows.append({
            "date": str(item.findtext("pubDate") or "")[:80],
            "title": title[:500],
            "url": link,
            "snippet": snippet,
            "source": source[:200],
        })
        if len(rows) >= max(1, min(int(max_results), 24)):
            break
    return rows

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
    """Search current news with bounded provider fallback.

    DDGS is attempted first.  If it yields no qualifying rows (or fails), one
    Google News RSS retrieval is used; sparse local daily requests broaden that
    fallback to one week.  All recovery stays inside this
    primitive, so the model never needs to churn near-duplicate ``news_search``
    calls.
    """
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
                "snippet": str(item.get("body", item.get("snippet", "")) or "")[:1200],
                "source": str(item.get("source", "") or "")[:200],
            })
        return compact

    completed_provider = False
    provider_errors: list[str] = []
    compact: list[dict] = []
    try:
        from ddgs import DDGS
        client = DDGS()
        try:
            results = _ddgs_news_rows(
                client, query=effective_query, region=region, timelimit=(window or None),
                max_results=max(limit * 2, limit + 4),
            )
            completed_provider = True
            compact = compact_rows(results)
            if location:
                compact = _location_scoped_news_rows(compact, location, limit=limit)
        except Exception as exc:
            provider_errors.append(f"DDGS: {exc}")

    except Exception as exc:
        provider_errors.append(f"DDGS unavailable: {exc}")

    if compact:
        return json.dumps(compact[:limit], ensure_ascii=False, indent=2)

    # Independent fallback: Google News RSS.  For a local daily request use the
    # same one-week ceiling as the sparse-index recovery; generic/topical news
    # keeps the user's requested window.
    rss_window = "w" if location and window in {"", "d"} else window
    try:
        rss_rows = _google_news_rss_rows(
            effective_query, region=region, timelimit=rss_window, max_results=max(limit * 2, limit + 4),
        )
        completed_provider = True
        compact = compact_rows(rss_rows)
        if location:
            compact = _location_scoped_news_rows(compact, location, limit=limit)
    except Exception as exc:
        provider_errors.append(f"Google News RSS: {exc}")

    if compact:
        return json.dumps(compact[:limit], ensure_ascii=False, indent=2)
    if completed_provider:
        return "[]"
    detail = "; ".join(provider_errors)[:700] or "all providers failed"
    return f"Error: news search failed: {detail}"


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


def _wiki_api_lookup(query: str) -> dict:
    """Fetch one Wikipedia result through the bounded MediaWiki JSON API."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 1,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
        "format": "json",
        "utf8": 1,
    }
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(params)
    _final_url, content_type, body = fetch_text(
        url,
        timeout=8.0,
        max_bytes=512 * 1024,
        allowed_types={"application/json", "text/json", "text/plain"},
    )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Wikipedia returned non-JSON content ({content_type or 'unknown'}).") from exc
    pages = payload.get("query", {}).get("pages", {}) if isinstance(payload, dict) else {}
    if not isinstance(pages, dict) or not pages:
        raise LookupError("Wikipedia returned no matching pages.")
    page = min(
        (item for item in pages.values() if isinstance(item, dict)),
        key=lambda item: int(item.get("index", 1_000_000)),
        default=None,
    )
    if not page:
        raise LookupError("Wikipedia returned no matching pages.")
    summary = " ".join(str(page.get("extract") or "").split())
    if not summary:
        raise LookupError("Wikipedia returned a page without a summary.")
    return {
        "title": str(page.get("title") or query)[:300],
        "url": str(page.get("fullurl") or "")[:1200],
        "summary": summary[:5000],
    }


def wiki_search(query: str = "") -> str:
    """Search Wikipedia for concise encyclopedic background information."""
    query = str(query).strip()
    if not query:
        return "Error: Missing required 'query' parameter."
    if len(query) > 1000:
        return "Error: Query is limited to 1000 characters."
    try:
        return json.dumps(_wiki_api_lookup(query), ensure_ascii=False, indent=2)
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


_BOILERPLATE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "form", "template", "svg"}
_BOILERPLATE_HINT_RE = re.compile(
    r"(?:^|[-_\s])(?:advert|ads?|banner|breadcrumb|cookie|footer|header|menu|nav|newsletter|promo|"
    r"recommend|related|share|sharing|sidebar|social|sponsor|subscribe|subscription|shopping|"
    r"carousel|gallery|comment|modal|popup)(?:$|[-_\s])",
    re.I,
)
_MAIN_CONTENT_SELECTORS = (
    "article",
    "main",
    "[role='main']",
    "#main-content",
    "#main_content",
    ".main-content",
    ".main_content",
    ".article-content",
    ".article-body",
    ".article__body",
    ".story-body",
    ".story-content",
    ".entry-content",
    ".post-content",
)


def _node_identity_text(node) -> str:
    # BeautifulSoup may leave descendants from a decomposed parent in a list
    # snapshot with ``attrs=None``. Treat those as already removed.
    if not getattr(node, "attrs", None):
        return ""
    classes = " ".join(str(item) for item in (node.get("class") or []))
    return f"{node.get('id') or ''} {classes}".strip()


def _remove_page_boilerplate(soup: BeautifulSoup) -> None:
    """Remove navigation/advertising chrome before article scoring.

    BeautifulSoup is intentionally kept as the only HTML dependency.  The old
    implementation removed semantic nav tags but then joined *all* remaining
    strings, which still admitted large recommendation, shopping, gallery, ad,
    and newsletter blocks on modern news sites.
    """
    for element in list(soup.find_all(_BOILERPLATE_TAGS)):
        element.decompose()
    for element in list(soup.find_all(True)):
        identity = _node_identity_text(element)
        if identity and _BOILERPLATE_HINT_RE.search(identity):
            element.decompose()


def _node_readable_text(node) -> str:
    """Render one candidate node as bounded human-readable block text."""
    blocks: list[str] = []
    for child in node.find_all(["h1", "h2", "h3", "p", "blockquote", "pre", "li"]):
        text = " ".join(child.stripped_strings)
        if text and (not blocks or blocks[-1] != text):
            blocks.append(text)
    if not blocks:
        raw = " ".join(node.stripped_strings)
        return re.sub(r"\s+", " ", raw).strip()
    return "\n".join(blocks)


def _content_score(node) -> tuple[float, str]:
    """Score an article/main candidate using prose density and link penalty."""
    text = _node_readable_text(node)
    if not text:
        return -1.0, ""
    text_len = len(text)
    paragraphs = [p for p in node.find_all("p") if len(" ".join(p.stripped_strings)) >= 40]
    sentences = len(re.findall(r"[.!?](?:\s|$)", text))
    link_chars = sum(len(" ".join(link.stripped_strings)) for link in node.find_all("a"))
    link_density = min(1.0, link_chars / max(1, text_len))
    # Prose-heavy nodes win; link farms and index pages lose heavily.
    score = (text_len * (1.0 - 0.85 * link_density)) + (len(paragraphs) * 180) + (sentences * 12)
    tag_name = str(getattr(node, "name", "") or "").lower()
    if tag_name in {"article", "main"} or str(node.get("role") or "").lower() == "main":
        score += 600
    return score, text


def extract_main_text(html: str, *, max_chars: int = 20000) -> str:
    """Extract the main article/document body from HTML using BeautifulSoup.

    Preference is given to semantic article/main containers.  If a page lacks
    those, sufficiently large ``section``/``div`` candidates are scored by prose
    density and link density.  Sparse/simple pages safely fall back to the
    cleaned body rather than returning nothing.
    """
    soup = BeautifulSoup(str(html or ""), "html.parser")
    _remove_page_boilerplate(soup)

    candidates = []
    seen: set[int] = set()
    for selector in _MAIN_CONTENT_SELECTORS:
        for node in soup.select(selector):
            if id(node) not in seen:
                seen.add(id(node))
                candidates.append(node)
    for node in soup.find_all(["section", "div"]):
        if id(node) in seen:
            continue
        # Avoid scoring every tiny layout wrapper.
        if len(" ".join(node.stripped_strings)) >= 300 and len(node.find_all("p")) >= 2:
            seen.add(id(node))
            candidates.append(node)

    best_text = ""
    best_score = -1.0
    for node in candidates:
        score, text = _content_score(node)
        if score > best_score:
            best_score, best_text = score, text

    if len(best_text) < 120:
        root = soup.body or soup
        best_text = _node_readable_text(root)
    return best_text[:max(500, int(max_chars))].strip()


def browse_url(url: str = "", extract: str = "", max_chars: int = 20000) -> str:
    """Fetch a public URL; optionally return only source passages relevant to an extraction instruction."""
    if not str(url).strip():
        return "Error: Missing required 'url' parameter."
    try:
        final_url, content_type, body = fetch_text(url, max_bytes=2 * 1024 * 1024)
        max_chars = max(1000, min(int(max_chars), 50000))
        if content_type in {"application/json", "application/xml", "text/xml", "text/plain"}:
            text = body[:50000]
        else:
            text = extract_main_text(body, max_chars=50000)
        extraction = " ".join(str(extract or "").split())
        if extraction:
            text = targeted_extract(text, extraction, max_chars=min(max_chars, 10000))
            mode = f"\nExtraction: {extraction[:500]}"
        else:
            text = text[:max_chars]
            mode = ""
        return f"URL: {final_url}\nContent-Type: {content_type}{mode}\n\n{text or 'The page returned no readable text content.'}"
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
    if canonical:
        return (
            f"I couldn't find any qualifying current local headlines for **{canonical}** in the news provider's bounded search. "
            "I won't substitute unrelated or wrong-location stories."
        )
    return (
        "I couldn't find any qualifying current headlines in the news provider's bounded search. "
        "I won't invent or substitute stale results."
    )


def format_news_provider_error(content: str, *, location: str = "") -> str:
    """Render one terminal provider failure for a news-only fact request."""
    canonical = canonicalize_location(location)
    detail = str(content or "").strip()
    detail = re.sub(r"^Error:\s*", "", detail, flags=re.I)[:240]
    suffix = f" ({detail})" if detail else ""
    if canonical:
        return f"I couldn't retrieve current local headlines for **{canonical}** because the news provider failed{suffix}."
    return f"I couldn't retrieve current headlines because the news provider failed{suffix}."


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
