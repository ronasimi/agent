"""Deterministic task requirements for broad multi-tool requests.

The ledger is intentionally conservative: it only creates requirements from
explicit phrases in the current user request.  It is used for tool exposure and
a pre-final completeness gate; it does not infer conclusions from tool output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .market import extract_market_instruments, is_market_price_request


@dataclass
class Requirement:
    key: str
    tool: str
    label: str
    status: str = "pending"
    attempts: int = 0
    last_reason: str = ""
    fingerprint: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tool": self.tool,
            "label": self.label,
            "status": self.status,
            "attempts": self.attempts,
            "last_reason": self.last_reason,
            "fingerprint": self.fingerprint,
            "scope": dict(self.scope),
            "evidence": [dict(row) for row in self.evidence if isinstance(row, dict)],
        }


_RULES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("weather_forecast", "weather_forecast", "current weather/forecast", (r"\b(?:weather|forecast|current conditions?)\b", r"\b(?:temperature|precipitation|rain|snow|humidity|wind speed)\b.*\b(?:today|tomorrow|current|now|tonight|week|days?|hours?)\b")),
    ("current_time", "current_time", "current clock time/date", (r"\bwhat time is it\b", r"\bcurrent time\b", r"\bcurrent date\b", r"\bwhat(?:'s| is) (?:today(?:'s)? date|the date)\b", r"\bwhat day is it\b", r"\b(?:local|utc) time\b", r"\bwhat timezone\b", r"\bcurrent timezone\b")),
    ("host_health", "host_snapshot", "host CPU/memory/disk/temperature state", (r"\bhost (?:health|cpu|memory|disk|temperature|state)", r"\bcpu,? memory,? disk", r"\b(?:host|system|cpu|gpu) temperature\b", r"\btemperature sensors?\b")),
    ("pressure", "pressure_snapshot", "CPU/memory/I/O pressure", (r"\bpressure (?:state|snapshot)?\b", r"\b(?:cpu|memory|i/o|io) pressure\b")),
    ("processes", "process_snapshot", "top resource-consuming processes", (r"\btop .*process", r"\bresource[- ]consuming process", r"\bprocess snapshot\b")),
    ("filesystem", "filesystem_snapshot", "filesystem capacity/inode state", (r"\bfilesystem", r"\binode", r"\bdisk capacity\b")),
    ("services", "service_health", "failed/unhealthy service state", (r"\b(?:failed|unhealthy) services?\b", r"\bservice health\b", r"\bservice warnings?\b")),
    ("network_state", "network_snapshot", "network interfaces/routes/listeners", (r"\bnetwork (?:interfaces|routes|health|state|status)\b", r"\blistening sockets?\b")),
    ("neighbors", "neighbor_snapshot", "network neighbor table", (r"\bnetwork neighbors?\b", r"\bnetwork\b[^.\n]{0,100}\bneighbors?\b", r"\bneighbor table\b", r"\barp(?: table)?\b", r"\bndp(?: table)?\b")),
    ("connections", "connection_snapshot", "established network connections", (r"\bestablished connections?\b", r"\bconnection snapshot\b", r"\bactive connections?\b")),
    ("local_subnets", "local_subnets", "active local private subnets", (
        r"\bscan (?:the )?local network\b",
        r"\bscan .*subnets?.*\bhosts?\b",
        r"\blocal network.*\bhosts?\b",
        r"\bdiscover .*\blocal (?:network|subnets?)\b",
    )),
    ("scan_subnet", "scan_subnet", "local subnet host discovery and fingerprinting", (
        r"\bscan (?:the )?local network\b",
        r"\bscan .*subnets?.*\bhosts?\b",
        r"\blocal network.*\bhosts?\b",
        r"\bdiscover .*\bhosts?\b.*\b(?:lan|subnet|local network)\b",
    )),
    ("dns", "dns_diagnose", "DNS resolution/diagnosis", (r"\bresolve\s+[a-z0-9.-]+", r"\bdns (?:resolution|diagnos|lookup)", r"\bresolve dns\b")),
    ("http", "http_probe", "HTTP/HTTPS connectivity probe", (
        r"\bprobe (?:https?|connectivity)",
        r"\bhttps? connectivity\b",
        r"\bhttp probe\b",
        r"\b(?:check|test)\s+(?:whether\s+)?https?://[^\s<>\"']+[^.\n]{0,80}\b(?:reachable|available|up|responding)\b",
        r"\bhttps?://[^\s<>\"']+[^.\n]{0,80}\b(?:reachable|available|up|responding)\b",
    )),
    ("read_file", "read_file", "requested file read", (
        r"\bread\s+(?:the\s+)?(?:file\s+)?(?:/|\./|\.\./)[^\s,;]+[^\n]{0,140}\b(?:summari[sz]e|inspect|show|report|if it exists)\b",
    )),
    ("path", "network_path", "network path/hop diagnosis", (r"\bnetwork path\b", r"\btraceroute\b", r"\bmtr\b", r"\broute tracing\b")),
    ("tool_health", "tool_health", "registered tool/dependency health", (r"\btool/?dependency health\b", r"\btool health\b", r"\bcurrent tool.*health\b")),
    ("dependency_audit", "dependency_audit", "runtime dependency audit", (r"\bdependency health\b", r"\bdependency audit\b", r"\btool/?dependency health\b")),
    ("news_search", "news_search", "current news headline discovery", (r"\b(?:latest|recent|current|today(?:'s)?)\b.{0,48}\b(?:news|headlines?|stories?)\b", r"\b(?:news|headlines?)\b.{0,48}\b(?:latest|recent|current|today)\b", r"\b(?:latest|top|local)\s+(?:news|headlines?)\b", r"^\s*(?:news|headlines?)\b")),
    ("market_quote", "market_quote", "current market/commodity quote", (r"\b(?:current|latest|live|today(?:'s)?|right now)\b.{0,64}\b(?:price|prices|quote|quotes|trading at)\b", r"\b(?:price|prices|quote|quotes)\b.{0,64}\b(?:wti|brent|crude oil|gold|silver|natural gas|copper)\b")),
    ("web_search", "web_search", "current web source discovery", (r"\bweb research\b", r"\bresearch the current\b", r"\bcurrent .*documentation\b", r"\blook up\b", r"\bsearch the web\b")),
    ("web_verify", "browse_url", "authoritative source content verification", (r"\bcurrent .*documentation\b", r"\bofficial .*documentation\b", r"\bsource urls?\b", r"\bverify .*source\b", r"\bweb research\b")),
    ("screenshot", "take_web_screenshot", "requested webpage screenshot", (r"\btake (?:a )?screenshot\b", r"\bscreenshot of\b", r"\bcapture .*page\b")),
    ("repo_status", "repo_status", "repository status", (r"\brepository status\b", r"\brepo status\b")),
    ("repo_checks", "repo_checks", "repository compile/config/lint/test checks", (r"\brepository health\b", r"\brepo checks?\b", r"\bcompile/config/lint/test\b", r"\b(?:compile|lint|pytest|tests?).*checks?\b")),
    ("gmail", "gmail_search_messages", "requested Gmail messages", (
        r"\b(?:check|search|show|find|list|read)\s+(?:through\s+)?my\s+(?:gmail|email|emails|mail|inbox|messages?)\b",
        r"\bwhat(?:'s| is)\s+in\s+my\s+(?:gmail|email|inbox)\b",
    )),
    ("google_calendar", "google_calendar_list_events", "requested Google Calendar schedule", (
        r"\b(?:check|search|show|list|read)\s+my\s+(?:google\s+)?calendar\b",
        r"\bwhat(?:'s| is)\s+on\s+my\s+(?:google\s+)?calendar\b",
        r"\bmy\s+(?:upcoming\s+)?(?:calendar\s+)?(?:events|meetings|appointments)\b",
        r"\b(?:what(?:'s| is)|show|check|list)\s+(?:on\s+)?my\s+schedule\b",
    )),
    ("google_drive", "google_drive_list_files", "requested Google Drive files", (
        r"\b(?:check|search|show|find|list|read)\s+(?:my\s+)?google\s+drive\b",
        r"\b(?:recent|recently modified|latest)\s+(?:google\s+)?drive\s+files?\b",
        r"\bgoogle\s+drive\b.{0,80}\bfiles?\b",
    )),
)

# Explicit tool names in the user's request are requirements as well.  This list
# is intentionally limited to read/diagnostic tools that are meaningful as
# completion checks; arbitrary mutating tools are not auto-required here.

# Some structured probes subsume narrower requirements.  This lets the ledger
# recognize verified equivalent evidence instead of forcing redundant calls.
_EQUIVALENT_REQUIREMENT_TOOLS: dict[str, tuple[str, ...]] = {
    "http_probe": ("dns_diagnose",),
    "endpoint_probe": ("dns_diagnose",),
}

_EXPLICIT_TOOL_NAMES = {
    "host_snapshot", "pressure_snapshot", "process_snapshot", "filesystem_snapshot",
    "service_health", "network_snapshot", "neighbor_snapshot", "connection_snapshot",
    "dns_diagnose", "network_path", "endpoint_probe", "http_probe", "tool_health",
    "dependency_audit", "news_search", "market_quote", "web_search", "browse_url", "take_web_screenshot", "geocode_location", "weather_forecast",
    "repo_status", "repo_checks", "page_metadata", "page_links", "extract_document", "read_file",
    "current_time", "hostname", "environment_summary", "local_subnets", "scan_subnet",
    "gmail_search_messages", "gmail_read_message", "google_calendar_list_events",
    "google_calendar_get_event", "google_calendar_list_calendars", "google_drive_list_files",
}

_FACT_RULE_INTENTS = {
    "weather_forecast": "weather",
    "current_time": "current_time",
    "host_health": "host_state",
    "network_state": "network_state",
    "news_search": "news",
    "market_quote": "market_price",
    "repo_status": "repository_state",
    "gmail": "gmail",
    "google_calendar": "google_calendar",
}

_TOOL_FACT_INTENTS = {
    "geocode_location": "weather",
    "weather_forecast": "weather",
    "news_search": "news",
    "market_quote": "market_price",
    "current_time": "current_time",
}


_IMPLEMENTATION_ACTION_RE = re.compile(
    r"\b(?:refactor|implement|debug|fix|patch|modify|change|update|write|create|build|test|review|"
    r"inspect|explain|compare|design|optimi[sz]e)\b",
    re.I,
)
_IMPLEMENTATION_ARTIFACT_RE = re.compile(
    r"\b(?:code|script|module|function|method|class|validator|formatter|parser|router|routing|"
    r"classifier|intent|prompt|regex|schema|widget|harness|application|app|ui|api|integration|"
    r"implementation|logic|library|package)\b",
    re.I,
)
_FACT_FEATURE_ARTIFACT_RE = re.compile(
    r"\b(?:weather|forecast|news|headlines?|current[ _-]?time)\s+"
    r"(?:api|validator|formatter|parser|router|classifier|tool|function|method|module|code|tests?|schema|widget|app)\b"
    r"|\b(?:api|validator|formatter|parser|router|classifier|tool|function|method|module|code|tests?|schema|widget|app)\s+"
    r"(?:for\s+)?(?:weather|forecast|news|headlines?|current[ _-]?time)\b",
    re.I,
)
_WEATHER_PRIMARY_RE = re.compile(r"\b(?:weather|forecast|current conditions?)\b", re.I)
_WEATHER_DETAIL_RE = re.compile(
    r"\b(?:temperature|precipitation|rain(?:fall|ing)?|snow(?:fall|ing)?|humidity|wind speed|"
    r"highs?|lows?|feels like|dew point)\b",
    re.I,
)
_REQUEST_CUE_RE = re.compile(
    r"(?:^\s*(?:what|when|where|will|is|are|show|check|get|give|tell|find|look up)\b|[?]\s*$)",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(today|tomorrow|tonight|now|current|this (?:morning|afternoon|evening|week|weekend)|next (?:\d+ )?(?:hours?|days?|week|weekend)|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.I,
)
_LOCATION_STOP = {"the", "weather", "forecast", "today", "tomorrow", "tonight", "now", "current", "currently", "please", "like"}

_CANADIAN_PROVINCES = {
    "ab": "Alberta", "bc": "British Columbia", "mb": "Manitoba", "nb": "New Brunswick",
    "nl": "Newfoundland and Labrador", "ns": "Nova Scotia", "nt": "Northwest Territories",
    "nu": "Nunavut", "on": "Ontario", "pe": "Prince Edward Island", "pei": "Prince Edward Island",
    "qc": "Quebec", "pq": "Quebec", "sk": "Saskatchewan", "yt": "Yukon",
}
_CANADIAN_PROVINCE_NAMES = {value.lower() for value in _CANADIAN_PROVINCES.values()}
_COUNTRY_ALIASES = {
    "uk": "United Kingdom", "u.k.": "United Kingdom", "usa": "United States",
    "u.s.": "United States", "us": "United States",
}
_NEWS_TIME_RE = re.compile(
    r"\b(?:latest|recent|current|today(?:'s)?|tonight|this\s+(?:week|weekend|month)|"
    r"last\s+(?:day|week|month)|past\s+\d+\s+(?:hours?|days?))\b",
    re.I,
)
_NEWS_GENERIC_TOKENS = {
    "a", "about", "are", "around", "current", "for", "from", "headlines", "headline", "in",
    "latest", "local", "me", "near", "news", "of", "please", "recent", "show", "stories",
    "story", "tell", "give", "get", "check", "find", "the", "today", "todays", "top", "what", "whats",
    # Request/formatting words are not news topics. Keeping these out of provider
    # queries is especially important for compound prompts such as
    # "retrieve exactly 3 ... for each report ...".
    "retrieve", "exactly", "each", "report", "reports", "return", "publisher",
    "publication", "date", "url", "urls", "duplicate", "duplicates",
}
# High-frequency news subjects that are commonly capitalized at the start of a
# sentence or written as acronyms.  They must not be mistaken for a city merely
# because the old parser used capitalization as a geographic heuristic.
_NEWS_TOPIC_HINTS = {
    "ai", "artificial intelligence", "business", "climate", "crypto", "cryptocurrency",
    "economy", "energy", "entertainment", "finance", "gaming", "health", "markets",
    "politics", "science", "sports", "tech", "technology", "world",
}


def is_implementation_request(user_text: str) -> bool:
    """Return whether fact-like words refer to software/harness work, not live facts.

    Small local models are especially vulnerable to lexical collisions such as
    ``refactor the weather validator`` versus ``check the weather``.  This gate
    requires an implementation action plus a concrete software artifact, while
    also recognizing direct compounds such as ``news API``.
    """
    text = " ".join(str(user_text or "").strip().split())
    if not text:
        return False
    return bool(
        _FACT_FEATURE_ARTIFACT_RE.search(text)
        or (_IMPLEMENTATION_ACTION_RE.search(text) and _IMPLEMENTATION_ARTIFACT_RE.search(text))
    )


def is_weather_fact_request(user_text: str) -> bool:
    """Distinguish a live weather request from code or conceptual discussion."""
    text = " ".join(str(user_text or "").strip().split())
    if not text or is_implementation_request(text):
        return False
    primary = bool(_WEATHER_PRIMARY_RE.search(text))
    detail = bool(_WEATHER_DETAIL_RE.search(text))
    temporal = bool(_TEMPORAL_RE.search(text))
    location_cue = bool(re.search(r"\b(?:in|at|near|for)\s+[A-Za-z]", text, re.I))
    direct = bool(_REQUEST_CUE_RE.search(text) or re.match(r"^\s*(?:weather|forecast)\b", text, re.I))
    live_detail_question = bool(re.search(
        r"\b(?:is it|will it|does it|what(?:'s| is) (?:the )?(?:temperature|humidity|wind speed))\b",
        text,
        re.I,
    ))
    return bool(
        (primary and (direct or temporal or location_cue))
        or (detail and direct and (temporal or location_cue or live_detail_question))
    )


def is_news_fact_request(user_text: str) -> bool:
    """Return whether the user is asking for current news/headline facts."""
    text = " ".join(str(user_text or "").strip().split())
    if not text or is_implementation_request(text):
        return False
    explicit = bool(re.search(
        r"(?:\b(?:latest|recent|current|today(?:'s)?)\b.{0,48}\b(?:news|headlines?|stories?)\b|"
        r"\b(?:news|headlines?)\b.{0,48}\b(?:latest|recent|current|today)\b|"
        r"\b(?:latest|top|local)\s+(?:news|headlines?)\b|"
        r"^\s*(?:news|headlines?)\b)",
        text,
        re.I,
    ))
    if explicit:
        return True
    # Compound direct requests may name the news noun without repeating a time
    # qualifier: "Give me today's weather and headlines about AI". The outer
    # request cue is enough once implementation/documentation prompts are ruled out.
    return bool(_REQUEST_CUE_RE.search(text) and re.search(r"\b(?:news|headlines?|stories?)\b", text, re.I))


def classify_request_intent(user_text: str) -> str:
    """Classify only explicit operational/live-fact intents for task framing."""
    text = " ".join(str(user_text or "").strip().split())
    lower = text.lower()
    implementation = is_implementation_request(text)
    if is_weather_fact_request(text):
        return "weather"
    if not implementation and re.search(r"\b(?:what time is it|current time|current date|what day is it|timezone)\b", lower):
        return "current_time"
    if not implementation and re.search(r"\b(?:host (?:health|cpu|memory|disk|temperature|state)|system health|host snapshot)\b", lower):
        return "host_state"
    if not implementation and re.search(r"\b(?:network (?:interfaces|routes|health|state|status|connections?)|neighbor table|arp table|ndp table)\b", lower):
        return "network_state"
    if not implementation and re.search(r"\b(?:repo(?:sitory)? (?:status|diff|health)|git status|git diff)\b", lower):
        return "repository_state"
    if not implementation and re.search(r"\bmy\s+(?:gmail|email|emails|mail|inbox|messages?)\b", lower):
        return "gmail"
    if not implementation and re.search(
        r"\b(?:my\s+(?:(?:google\s+)?calendar|schedule|events|meetings|appointments)|"
        r"(?:google\s+)?calendar\s+events?)\b",
        lower,
    ):
        return "google_calendar"
    if is_news_fact_request(text):
        return "news"
    if not implementation and is_market_price_request(text):
        return "market_price"
    return ""

# Temporal qualifiers are deliberately stripped before a weather phrase is
# allowed to become a location candidate.  This prevents prompts such as
# "weather right now" from geocoding "right" (which can resolve to a real
# place) instead of falling back to the user's declared location.
_WEATHER_TEMPORAL_QUALIFIER_RE = re.compile(
    r"(?:"
    r"\bright\s+now\b|"
    r"\bat\s+the\s+moment\b|\bthe\s+moment\b|\bat\s+present\b|\bpresent\b|"
    r"\bcurrently\b|\bcurrent(?:ly)?\b|"
    r"\b(?:today|tomorrow|tonight|now)\b|"
    r"\bthis\s+(?:morning|afternoon|evening|week|weekend)\b|"
    r"\bnext\s+(?:\d+\s+)?(?:hours?|days?|week|weekend)\b|"
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b"
    r")",
    re.I,
)


def _clean_entity(value: str) -> str:
    text = re.sub(r"[?!.;,]+$", "", str(value or "").strip())
    match = _WEATHER_TEMPORAL_QUALIFIER_RE.search(text)
    if match:
        text = text[:match.start()]
    return re.sub(r"\s+", " ", text).strip(" ,:-")[:180]


def canonicalize_location(value: str) -> str:
    """Expand high-value country/province abbreviations for unambiguous search.

    This intentionally avoids guessing arbitrary two-letter tokens as US states;
    short Canadian province names are expanded because the configured/default
    news region is Canadian and prompts such as ``London ON`` otherwise drift to
    London, England.
    """
    raw = re.sub(r"\s+", " ", str(value or "")).strip(" ,:-")
    if not raw:
        return ""
    raw = re.sub(r"[?!.;]+$", "", raw).strip()
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    words = raw.replace(",", " ").split()
    if words:
        last = words[-1].lower().rstrip(".")
        if last in _CANADIAN_PROVINCES:
            city = " ".join(words[:-1]).strip(" ,")
            province = _CANADIAN_PROVINCES[last]
            return ", ".join(part for part in (city, province, "Canada") if part)[:180]
        if last in _COUNTRY_ALIASES:
            place = " ".join(words[:-1]).strip(" ,")
            return ", ".join(part for part in (place, _COUNTRY_ALIASES[last]) if part)[:180]
    lower = raw.lower()
    if "canada" not in lower:
        matched_province = next((name for name in _CANADIAN_PROVINCE_NAMES if re.search(rf"\b{re.escape(name)}\b", lower)), "")
        if matched_province:
            pieces = pieces or [raw]
            pieces.append("Canada")
            return ", ".join(dict.fromkeys(pieces))[:180]
    return raw[:180]


def _clean_news_entity(value: str) -> str:
    text = re.sub(r"[?!.;]+$", "", str(value or "").strip())
    match = _NEWS_TIME_RE.search(text)
    if match:
        text = text[:match.start()]
    text = re.sub(r"\b(?:local\s+)?(?:news|headlines?|stories?)\b.*$", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,:-")
    return canonicalize_location(text)


def _looks_like_location_candidate(value: str) -> bool:
    raw = str(value or "").strip(" ,")
    if not raw or len(raw.split()) > 8:
        return False
    normalized = re.sub(r"\s+", " ", raw).strip().lower()
    lower_words = [word.lower().rstrip(".") for word in re.findall(r"[A-Za-z]+", raw)]
    if lower_words and lower_words[0] in {"give", "show", "tell", "what", "whats", "get", "find", "check", "latest", "recent", "current", "today", "todays", "top", "local"}:
        return False
    if normalized in _NEWS_TOPIC_HINTS:
        return False
    if any(word in _CANADIAN_PROVINCES or word in _COUNTRY_ALIASES for word in lower_words):
        return True
    if any(name in normalized for name in _CANADIAN_PROVINCE_NAMES) or "," in raw:
        return True
    # Acronyms such as AI are overwhelmingly more likely to be topics than
    # locations.  For ordinary proper names retain the useful lightweight
    # heuristic (Toronto news, New York headlines) while the topic guard above
    # catches common title-cased subjects such as Business or Technology.
    if raw.isupper() and len(raw) <= 6:
        return False
    return bool(re.search(r"(?:^|\s)[A-Z][A-Za-z'-]+", raw))


def _extract_news_entity(
    text: str,
    previous: dict[str, Any] | None = None,
    default_location: str = "",
) -> str:
    value = " ".join(str(text or "").split())
    # Location after the news noun: "headlines in London ON".
    match = re.search(
        r"\b(?:news|headlines?|stories?)\b[^?]{0,80}?\b(?:in|for|from|near|around)\s+([^?]+?)[?!.]*$",
        value,
        flags=re.I,
    )
    if match and _looks_like_location_candidate(match.group(1)):
        candidate = _clean_news_entity(match.group(1))
        if candidate:
            return candidate
    # Explicit local location before the noun: "latest 3 local London, Ontario headlines".
    match = re.search(r"\blocal\s+(.{1,100}?)\s+(?:news|headlines?|stories?)\b", value, re.I)
    if match and _looks_like_location_candidate(match.group(1)):
        candidate = _clean_news_entity(match.group(1))
        if candidate:
            return candidate
    # Location before the noun: "London ON local news".
    match = re.search(r"^(?:what(?:'s| is| are)?\s+)?(?:the\s+)?(.{1,100}?)\s+(?:local\s+)?(?:news|headlines?)\b", value, re.I)
    if match:
        raw = re.sub(r"^(?:latest|recent|current|today(?:'s)?|top)\s+", "", match.group(1), flags=re.I)
        if _looks_like_location_candidate(raw):
            candidate = _clean_news_entity(raw)
            if candidate:
                return candidate
    lower = value.lower()
    if previous and str(previous.get("intent") or "") == "news":
        # Carry a prior locality only when the current turn is explicitly
        # deictic/referential.  A complete generic request such as "latest
        # headlines" starts a fresh general-news scope instead of inheriting
        # London (or any other prior city).
        inherit_location = bool(
            re.search(r"\b(?:local|nearby|near me|around me|there|same (?:place|area|city|location))\b", lower)
            or re.match(r"^(?:what about|and|also)\b", lower)
        )
        if inherit_location:
            prior = str(previous.get("entity") or "").strip()
            if prior:
                return prior
    if re.search(r"\b(?:local|nearby|near me|around me)\b", lower):
        return canonicalize_location(default_location)
    return ""


def _news_topic_terms(text: str, entity: str = "") -> list[str]:
    entity_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", str(entity or ""))}
    result: list[str] = []
    for token in re.findall(r"[A-Za-z0-9'-]+", str(text or "")):
        lower = token.lower().replace("'", "")
        if lower in _NEWS_GENERIC_TOKENS or lower in entity_tokens or len(lower) < 2:
            continue
        if lower not in result:
            result.append(lower)
    return result[:8]


def build_news_query(user_text: str, frame: dict[str, Any] | None = None, default_location: str = "") -> str:
    """Build a compact news query without inventing geographic scope.

    ``default_location`` is used only for an explicitly local/deictic request.
    A plain request such as ``latest headlines`` is intentionally general even
    when the user profile contains a home city.
    """
    resolved = dict(frame or {})
    raw = " ".join(str(resolved.get("source_text") or user_text or "").strip().split())
    entity = canonicalize_location(str(resolved.get("entity") or ""))
    if not entity and re.search(r"\b(?:local|nearby|near me|around me)\b", raw, re.I):
        entity = canonicalize_location(default_location)
    topics = _news_topic_terms(raw, entity)
    time_scope = str(resolved.get("time_scope") or "latest").strip().lower()
    if time_scope in {"", "current", "recent"}:
        time_scope = "latest"
    topical = (" " + " ".join(topics)) if topics else ""
    if entity:
        return f"{entity} local{topical} {time_scope} news"[:1000]
    if topics:
        return f"{' '.join(topics)} {time_scope} news"[:1000]
    return f"{time_scope} news"[:1000]


def news_region_for_frame(frame: dict[str, Any] | None = None, default_location: str = "") -> str:
    location = canonicalize_location(str((frame or {}).get("entity") or default_location or "")).lower()
    if "canada" in location or any(name in location for name in _CANADIAN_PROVINCE_NAMES):
        return "ca-en"
    if any(token in location for token in ("united kingdom", "england", "scotland", "wales", "northern ireland")):
        return "uk-en"
    if "australia" in location:
        return "au-en"
    return "ca-en"


def _extract_weather_entity(text: str, previous: dict[str, Any] | None = None) -> str:
    value = " ".join(str(text or "").split())
    # Prefer explicit prepositional location phrases.
    matches = list(re.finditer(r"\b(?:in|at|near)\s+([^?]+)", value, flags=re.I))
    if matches:
        candidate = _clean_entity(matches[-1].group(1))
        if candidate:
            return candidate
    # "weather for Toronto tomorrow" and similar.
    match = re.search(
        r"\b(?:weather|forecast)\s+(?:for\s+)?(.+?)(?:[.!?](?:\s|$)|$)",
        value, flags=re.I,
    )
    if match:
        candidate = _clean_entity(match.group(1))
        words = [w for w in re.findall(r"[A-Za-z0-9'-]+", candidate) if w.lower() not in _LOCATION_STOP]
        if words:
            return " ".join(words[:8])
    # Referential fragment: "And Toronto?"
    if previous and str(previous.get("intent") or "") == "weather":
        fragment = re.sub(r"^(?:and|also|what about)\s+", "", value, flags=re.I).strip()
        candidate = _clean_entity(fragment)
        generic = {"weather", "forecast", "the weather", "the forecast"}
        if candidate and candidate.lower() not in generic and not _TEMPORAL_RE.fullmatch(candidate) and len(candidate.split()) <= 8:
            if not re.search(r"\b(?:write|create|run|show|explain|tell|make|code|script)\b", candidate, re.I):
                return candidate
    return str((previous or {}).get("entity") or "")


def _extract_time_scope(text: str, previous: dict[str, Any] | None = None) -> str:
    match = _TEMPORAL_RE.search(str(text or ""))
    if match:
        return re.sub(r"\s+", " ", match.group(1).lower())
    return str((previous or {}).get("time_scope") or "")


def derive_task_frame(
    user_text: str,
    previous_frame: dict[str, Any] | None = None,
    default_location: str = "",
) -> dict[str, Any]:
    """Resolve the current task intent/entity/time scope for referential follow-ups."""
    text = " ".join(str(user_text or "").strip().split())
    previous = dict(previous_frame or {})
    intent = classify_request_intent(text)
    if not intent and is_followup_request(text) and previous.get("intent"):
        intent = str(previous.get("intent"))

    frame: dict[str, Any] = {"intent": intent}
    if intent == "weather":
        frame["entity"] = _extract_weather_entity(text, previous) or canonicalize_location(default_location)
        frame["time_scope"] = _extract_time_scope(text, previous) or "current"
    elif intent == "news":
        frame["entity"] = _extract_news_entity(text, previous, default_location)
        match = _NEWS_TIME_RE.search(text)
        frame["time_scope"] = re.sub(r"\s+", " ", match.group(0).lower()) if match else "current"
    elif intent == "current_time":
        tz = re.search(r"\b(?:in|for)\s+([A-Za-z][A-Za-z0-9_+:/ .-]{0,80}?)[?!.]*$", text)
        if tz:
            frame["entity"] = _clean_entity(tz.group(1))
        elif str(previous.get("intent") or "") == "current_time":
            frame["entity"] = str(previous.get("entity") or "")
    elif intent == "market_price":
        instruments = extract_market_instruments(text)
        if not instruments and str(previous.get("intent") or "") == "market_price":
            instruments = [str(item) for item in (previous.get("instruments") or []) if str(item)]
        if instruments:
            frame["instruments"] = instruments
        frame["time_scope"] = "current"
    return {key: value for key, value in frame.items() if value not in ("", None)}


@dataclass(frozen=True)
class FactSpan:
    fact_type: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ParsedModifier:
    kind: str
    value: str
    start: int
    end: int
    scope: str = "local"


_FACT_FRAME_ANCHORS: dict[str, re.Pattern[str]] = {
    "weather": re.compile(r"\b(?:weather|forecast|current conditions?|temperature|precipitation|rain|snow|humidity|wind speed)\b", re.I),
    "news": re.compile(r"\b(?:news|headlines?|stories?)\b", re.I),
    "current_time": re.compile(r"\b(?:current time|current date|what time|what day|local time|utc time|timezone|time is it)\b", re.I),
    "host_state": re.compile(r"\b(?:host (?:health|cpu|memory|disk|temperature|state)|system health|host snapshot)\b", re.I),
    "network_state": re.compile(r"\b(?:network (?:interfaces|routes|health|state|status|connections?)|neighbor table|arp table|ndp table)\b", re.I),
    "repository_state": re.compile(r"\b(?:repo(?:sitory)? (?:status|diff|health)|git status|git diff)\b", re.I),
    "market_price": re.compile(r"\b(?:price|prices|quote|quotes|trading at|worth|wti|brent|crude oil|gold|silver|natural gas|copper)\b", re.I),
}
_COMPOUND_CONNECTOR_RE = re.compile(r"\b(?:and|as\s+well\s+as|along\s+with|plus)\b|;", re.I)
_ALL_TIME_MODIFIER_RE = re.compile(
    r"\b(?:latest|recent|current|today(?:'s)?|tomorrow|tonight|now|right\s+now|"
    r"this\s+(?:morning|afternoon|evening|week|weekend|month)|next\s+(?:\d+\s+)?(?:hours?|days?|week|weekend)|"
    r"last\s+(?:day|week|month)|past\s+\d+\s+(?:hours?|days?)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)


def detect_fact_frame_types(user_text: str) -> set[str]:
    """Detect independently scoped live-fact intents without choosing a primary one."""
    raw = str(user_text or "")
    text = " ".join(raw.strip().split())
    if not text:
        return set()
    tool_recipe_requirements = derive_tool_recipe_stress_requirements(raw)
    if tool_recipe_requirements:
        return {
            str((item.scope or {}).get("fact_type") or "")
            for item in tool_recipe_requirements
            if str((item.scope or {}).get("fact_type") or "")
        }
    stress_requirements = derive_stress_requirements(raw)
    if stress_requirements:
        return {
            str((item.scope or {}).get("fact_type") or "")
            for item in stress_requirements
            if str((item.scope or {}).get("fact_type") or "")
        }
    if is_implementation_request(text):
        return set()
    result: set[str] = set()
    if is_weather_fact_request(text):
        result.add("weather")
    if is_news_fact_request(text):
        result.add("news")
    lower = text.lower()
    if re.search(r"\b(?:what time is it|current time|current date|what day is it|local time|utc time|timezone)\b", lower):
        result.add("current_time")
    if re.search(r"\b(?:host (?:health|cpu|memory|disk|temperature|state)|system health|host snapshot)\b", lower):
        result.add("host_state")
    if re.search(r"\b(?:network (?:interfaces|routes|health|state|status|connections?)|neighbor table|arp table|ndp table)\b", lower):
        result.add("network_state")
    if re.search(r"\b(?:repo(?:sitory)? (?:status|diff|health)|git status|git diff)\b", lower):
        result.add("repository_state")
    if is_market_price_request(text):
        result.add("market_price")
    return result


def _first_fact_anchor(text: str, fact_type: str) -> re.Match[str] | None:
    pattern = _FACT_FRAME_ANCHORS.get(str(fact_type or ""))
    return pattern.search(text) if pattern is not None else None


def _compound_clause_ranges(text: str, fact_types: set[str]) -> list[tuple[int, int]]:
    """Split only between adjacent *different* fact anchors.

    The cut is placed at the connector nearest the following fact anchor. This
    intentionally keeps entity conjunctions such as ``weather in London and
    Windsor`` intact while splitting ``headlines and weather``.
    """
    if len(fact_types) < 2:
        return [(0, len(text))]
    anchors: list[tuple[int, int, str]] = []
    for fact_type in fact_types:
        match = _first_fact_anchor(text, fact_type)
        if match:
            anchors.append((match.start(), match.end(), fact_type))
    anchors.sort()
    cuts: list[tuple[int, int]] = []
    for left, right in zip(anchors, anchors[1:]):
        if left[2] == right[2]:
            continue
        connectors = list(_COMPOUND_CONNECTOR_RE.finditer(text, left[1], right[0]))
        if connectors:
            chosen = connectors[-1]
            cuts.append((chosen.start(), chosen.end()))
            continue
        # Commas may coordinate independent fact nouns, but never split a comma
        # that is merely part of a place name unless distinct anchors surround it.
        comma = text.rfind(",", left[1], right[0])
        if comma >= 0:
            cuts.append((comma, comma + 1))
    if not cuts:
        return [(0, len(text))]
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(set(cuts)):
        if start > cursor:
            ranges.append((cursor, start))
        cursor = end
    if cursor < len(text):
        ranges.append((cursor, len(text)))
    return [(start, end) for start, end in ranges if text[start:end].strip()] or [(0, len(text))]


_STRUCTURED_ITEM_RE = re.compile(r"(?m)^[ \t]*(?:\d{1,3}[.)]|[-*])\s+")


def _normalize_fact_source(user_text: str) -> str:
    """Normalize horizontal whitespace while preserving task/list boundaries."""
    text = str(user_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    return "\n".join(lines)


def _structured_item_ranges(text: str) -> list[tuple[int, int]]:
    """Return content ranges for numbered/bulleted task items.

    Markers are excluded so entity extraction never consumes the next item
    number (for example ``London, Ontario. 2. Get the latest...``).
    """
    matches = list(_STRUCTURED_ITEM_RE.finditer(text))
    if not matches:
        return []
    ranges: list[tuple[int, int]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        while end > start and text[end - 1].isspace():
            end -= 1
        if text[start:end].strip():
            ranges.append((start, end))
    return ranges


def _fact_clause(text: str, fact_type: str, fact_types: set[str]) -> FactSpan:
    anchor = _first_fact_anchor(text, fact_type)
    if anchor is None:
        return FactSpan(fact_type, 0, len(text), text.strip())

    # Numbered/bulleted requests are authoritative clause boundaries.  Resolve
    # the fact within its own item before considering conjunction punctuation.
    # This prevents item 1's location from swallowing item 2 and prevents later
    # instructions from becoming part of news/market source text.
    for item_start, item_end in _structured_item_ranges(text):
        if item_start <= anchor.start() < item_end:
            item_text = text[item_start:item_end]
            local_types = {
                name for name in fact_types
                if (match := _first_fact_anchor(item_text, name)) is not None
            }
            if len(local_types) <= 1:
                source = item_text.strip(" \t\n,;:-")
                offset = item_text.find(source) if source else 0
                return FactSpan(fact_type, item_start + max(0, offset), item_start + max(0, offset) + len(source), source)
            local_anchor = _first_fact_anchor(item_text, fact_type)
            if local_anchor is not None:
                for local_start, local_end in _compound_clause_ranges(item_text, local_types):
                    if local_start <= local_anchor.start() < local_end:
                        raw = item_text[local_start:local_end]
                        source = raw.strip(" \t\n,;:-")
                        offset = raw.find(source) if source else 0
                        absolute = item_start + local_start + max(0, offset)
                        return FactSpan(fact_type, absolute, absolute + len(source), source)

    for range_start, range_end in _compound_clause_ranges(text, fact_types):
        if range_start <= anchor.start() < range_end:
            source = text[range_start:range_end].strip(" \t\n,;:-")
            offset = text[range_start:range_end].find(source) if source else 0
            return FactSpan(fact_type, range_start + max(0, offset), range_start + max(0, offset) + len(source), source)
    return FactSpan(fact_type, anchor.start(), anchor.end(), anchor.group(0))


def _shared_time_modifier(text: str, fact_types: set[str]) -> ParsedModifier | None:
    anchors = [m for fact_type in fact_types if (m := _first_fact_anchor(text, fact_type)) is not None]
    if not anchors:
        return None
    first_anchor = min(match.start() for match in anchors)
    matches = [m for m in _ALL_TIME_MODIFIER_RE.finditer(text) if m.start() <= first_anchor]
    if not matches:
        return None
    chosen = matches[-1]
    return ParsedModifier("time", re.sub(r"\s+", " ", chosen.group(0).lower()), chosen.start(), chosen.end(), "shared")


def _local_time_scope(source_text: str, fact_type: str) -> str:
    if fact_type == "news":
        match = _NEWS_TIME_RE.search(source_text)
    else:
        match = _TEMPORAL_RE.search(source_text) or _ALL_TIME_MODIFIER_RE.search(source_text)
    return re.sub(r"\s+", " ", match.group(0).lower()) if match else ""


def derive_fact_frames(
    user_text: str,
    previous_frames: dict[str, dict[str, Any]] | None = None,
    *,
    default_location: str = "",
    required_fact_types: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return one independently scoped frame for every requested fact type.

    Required fact detection precedes clause decomposition. This prevents ordinary
    entity conjunctions from being mistaken for multiple intents and lets shared
    leading modifiers (for example ``today's``) propagate to coordinated facts.
    """
    text = _normalize_fact_source(user_text)
    previous = {str(k): dict(v or {}) for k, v in dict(previous_frames or {}).items() if isinstance(v, dict)}
    fact_types = set(required_fact_types or detect_fact_frame_types(text))
    if not fact_types:
        return {}
    shared_time = _shared_time_modifier(text, fact_types)
    frames: dict[str, dict[str, Any]] = {}
    for fact_type in sorted(fact_types):
        span = _fact_clause(text, fact_type, fact_types)
        prior = dict(previous.get(fact_type) or {})
        source = span.text or text
        frame: dict[str, Any] = {
            "intent": fact_type,
            "source_text": source,
            "source_span": [span.start, span.end],
        }
        if fact_type == "weather":
            frame["entity"] = _extract_weather_entity(source, prior) or canonicalize_location(default_location)
            frame["time_scope"] = _local_time_scope(source, fact_type) or (shared_time.value if shared_time else "") or str(prior.get("time_scope") or "") or "current"
        elif fact_type == "news":
            frame["entity"] = _extract_news_entity(source, prior, default_location)
            frame["time_scope"] = _local_time_scope(source, fact_type) or (shared_time.value if shared_time else "") or "current"
        elif fact_type == "current_time":
            tz = re.search(r"\b(?:in|for)\s+([A-Za-z][A-Za-z0-9_+:/ .-]{0,80}?)[?!.]*$", source)
            if tz:
                frame["entity"] = _clean_entity(tz.group(1))
            elif prior.get("entity"):
                frame["entity"] = str(prior.get("entity") or "")
            frame["time_scope"] = _local_time_scope(source, fact_type) or (shared_time.value if shared_time else "") or "current"
        elif fact_type == "market_price":
            instruments = extract_market_instruments(text)
            if not instruments:
                instruments = [str(item) for item in (prior.get("instruments") or []) if str(item)]
            if instruments:
                frame["instruments"] = instruments
            frame["time_scope"] = _local_time_scope(source, fact_type) or (shared_time.value if shared_time else "") or "current"
        else:
            local_time = _local_time_scope(source, fact_type)
            if local_time or shared_time:
                frame["time_scope"] = local_time or str(shared_time.value)
        frames[fact_type] = {key: value for key, value in frame.items() if value not in ("", None, [])}
    # Completeness invariant: a detected/required fact can be sparse, but may not
    # silently disappear during parsing.
    for fact_type in fact_types:
        frames.setdefault(fact_type, {"intent": fact_type, "source_text": text, "source_span": [0, len(text)]})
    return frames


def select_primary_fact_frame(
    fact_frames: dict[str, dict[str, Any]] | None,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the compatibility ``task_frame`` projection for legacy callers."""
    frames = {str(k): dict(v or {}) for k, v in dict(fact_frames or {}).items() if isinstance(v, dict)}
    legacy = dict(fallback or {})
    preferred = str(legacy.get("intent") or "")
    if preferred in frames:
        return dict(frames[preferred])
    if frames:
        return dict(min(frames.values(), key=lambda item: int((item.get("source_span") or [10**9])[0])))
    return legacy


def is_task_continuation(user_text: str, previous_frame: dict[str, Any] | None = None) -> bool:
    """Detect both referential follow-ups and underspecified same-intent turns.

    ``latest local headlines`` is a complete news sentence but ``local`` is
    deictic: it must inherit the previous news location.  Conversely, a topical
    request such as ``latest AI news`` starts a new frame and must not inherit a
    city merely because it also contains the word ``news``.
    """
    if is_followup_request(user_text):
        return True
    previous = dict(previous_frame or {})
    previous_intent = str(previous.get("intent") or "")
    if not previous_intent:
        return False
    current = derive_task_frame(user_text)
    if str(current.get("intent") or "") != previous_intent:
        return False
    if previous_intent == "news" and previous.get("entity") and not current.get("entity"):
        lower = str(user_text or "").lower()
        return bool(re.search(
            r"\b(?:local|nearby|near me|around me|there|same (?:place|area|city|location))\b",
            lower,
        ))
    if previous_intent == "weather" and previous.get("entity") and not current.get("entity"):
        return True
    if previous_intent == "current_time" and previous.get("entity") and not current.get("entity"):
        return True
    if previous_intent == "market_price" and previous.get("instruments") and not current.get("instruments"):
        return True
    return False


def effective_request_for_frame(user_text: str, frame: dict[str, Any] | None) -> str:
    frame = dict(frame or {})
    if not frame.get("intent"):
        return str(user_text or "")
    parts = [str(frame.get("intent") or "")]
    if frame.get("entity"):
        parts.append(str(frame["entity"]))
    if frame.get("time_scope"):
        parts.append(str(frame["time_scope"]))
    # Prefer the fact-specific source span for compound requests.  Re-appending
    # the entire original prompt pollutes recovery queries with unrelated list
    # items and policy prose (for example a weather geocode query containing
    # news, market, network, and file instructions).
    parts.append(str(frame.get("source_text") or user_text or ""))
    return " ".join(parts)


def _extract_target(tool: str, text: str) -> str:
    lower = str(text or "")
    if tool == "dns_diagnose":
        match = re.search(r"\bresolve\s+([a-z0-9.-]+)", lower, re.I) or re.search(r"\bdns(?: lookup| resolution| diagnose)?\s+(?:for\s+)?([a-z0-9.-]+)", lower, re.I)
        return str(match.group(1)).lower().rstrip(".") if match else ""
    if tool in {"http_probe", "browse_url"}:
        match = re.search(r"https?://[^\s<>'\"]+", lower, re.I)
        return match.group(0).rstrip(".,)") if match else ""
    if tool == "read_file":
        match = re.search(r"\bread\s+(?:the\s+)?(?:file\s+)?((?:/|\./|\.\./)[^\s,;]+)", lower, re.I)
        return match.group(1).rstrip(".,)") if match else ""
    if tool == "network_path":
        match = re.search(r"\b(?:traceroute|mtr|network path(?: to)?)\s+([a-z0-9.:-]+)", lower, re.I)
        return str(match.group(1)).lower() if match else ""
    return ""



_STRESS_SECTION_NAMES = (
    "REMOTE / INTERNET TOOLS",
    "SYSTEM TOOLS",
    "HOST TOOLS",
    "GOOGLE ACCOUNT TOOLS",
    "NETWORK TOOLS",
    "CROSS-CAPABILITY CONSISTENCY TESTS",
)
_STRESS_SECTION_RE = re.compile(
    r"(?mi)^\s*(REMOTE / INTERNET TOOLS|SYSTEM TOOLS|HOST TOOLS|GOOGLE ACCOUNT TOOLS|NETWORK TOOLS|CROSS-CAPABILITY CONSISTENCY TESTS)\s*$"
)
_NUMBERED_LINE_RE = re.compile(r"(?m)^\s*(\d{1,3})\.\s+")

_TOOL_RECIPE_STRESS_SECTION_RE = re.compile(
    r"(?mi)^\s*([A-L])\.\s*(DIRECT TOOL ROUTING|SIMILAR-TOOL DISAMBIGUATION|"
    r"TOOL DISCOVERY / RECOVERY PATH|WORKSPACE PATH AND FILE-TOOL TEST|"
    r"RECIPE DISCOVERY AND DUPLICATION CHECK|RECIPE CREATION|RECIPE REPLAY|"
    r"RECIPE FAILURE / FALLBACK BEHAVIOR|OBSERVATION / TRUNCATION PATH|"
    r"RECIPE STORAGE INTEGRITY|CLEANUP|REQUIREMENT / EVIDENCE AUDIT)\s*$"
)

_GENERALIZED_RECIPE_STRESS_SECTION_RE = re.compile(
    r"(?mi)^\s*([A-N])\.\s*(LEDGER AND SYSTEM BASELINE|TOOL LAYER SEPARATION|"
    r"CAPABILITY DISCOVERY AND PROVENANCE|DISPOSABLE WORKSPACE TEST|"
    r"SEMANTIC RECIPE SEARCH|GENERALIZED RECIPE CREATION|FIRST PARAMETERIZED REPLAY|"
    r"SECOND PARAMETERIZED REPLAY|GENERALIZATION AUDIT|ROUTING AND DISCOVERY AUDIT|"
    r"OBSERVATION AND FAILURE AUDIT|WORKSPACE CLEANUP|PERSISTED LEDGER AUDIT|FINAL AUDITS)\s*$"
)
_GENERALIZED_RECIPE_GENERAL_RULES_RE = re.compile(
    r"(?mis)^\s*GENERAL RULES\s*$([\s\S]*?)(?=^\s*=+\s*$\n\s*A\.\s*LEDGER AND SYSTEM BASELINE\s*$)"
)


def _generalized_recipe_numbered_items(user_text: str) -> list[tuple[int, str, str]]:
    """Parse the generalized parameterized-recipe stress plan.

    Unlike the earlier 37-item test, this contract explicitly numbers its
    GENERAL RULES as requirements 1-15 and then continues with executable and
    audit requirements 16-72. Recognition is structural rather than tied to the
    exact opening sentence so small wording changes do not drop the plan back to
    the generic requirement extractor.
    """
    text = str(user_text or "")
    headings = list(_GENERALIZED_RECIPE_STRESS_SECTION_RE.finditer(text))
    if len(headings) < 10:
        return []
    lower = text.lower()
    structural_markers = (
        "parameterized replay", "persisted ledger audit",
        "generalized recipe creation", "tool_search provenance",
    )
    if sum(marker in lower for marker in structural_markers) < 3:
        return []

    items: list[tuple[int, str, str]] = []
    rules = _GENERALIZED_RECIPE_GENERAL_RULES_RE.search(text)
    if rules:
        block = rules.group(1)
        matches = list(_NUMBERED_LINE_RE.finditer(block))
        for idx, match in enumerate(matches):
            item_start = match.end()
            item_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
            body = block[item_start:item_end].strip()
            if body:
                items.append((int(match.group(1)), "GENERAL RULES", body))

    for idx, heading in enumerate(headings):
        section = f"{heading.group(1).upper()}. {heading.group(2).strip()}"
        start = heading.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        final_output = re.search(r"(?mi)^\s*FINAL OUTPUT\s*$", text[start:end])
        if final_output:
            end = start + final_output.start()
        block = text[start:end]
        matches = list(_NUMBERED_LINE_RE.finditer(block))
        for j, match in enumerate(matches):
            item_start = match.end()
            item_end = matches[j + 1].start() if j + 1 < len(matches) else len(block)
            body = block[item_start:item_end].strip()
            if body:
                items.append((int(match.group(1)), section, body))

    by_number: dict[int, tuple[int, str, str]] = {}
    for row in items:
        by_number.setdefault(row[0], row)
    ordered = [by_number[number] for number in sorted(by_number)]
    # This deterministic plan is intentionally selected only for the complete
    # contract. Partial/ad-hoc prompts continue through the ordinary compiler.
    if [row[0] for row in ordered] != list(range(1, 73)):
        return []
    return ordered


def _generalized_recipe_requirement(number: int, section: str, body: str) -> Requirement:
    scope: dict[str, Any] = {
        "item_number": int(number),
        "section": section,
        "source_text": body[:2400],
        "generalized_recipe_stress": True,
    }
    key = f"genrecipe:{number:02d}"

    def req(tool: str, label: str, *, derived: bool = False, **extra: Any) -> Requirement:
        scope.update({k: v for k, v in extra.items() if v not in (None, "", [])})
        if derived:
            scope["derived"] = True
        return Requirement(key=key, tool=tool, label=label, scope=dict(scope))

    # Rules 1-15 are first-class requirements in this contract. They are closed
    # by deterministic policy/mutation/retry audits after execution.
    if 1 <= number <= 15:
        return req(f"__genrecipe_rule_{number:02d}__", f"general rule {number}", derived=True, policy_rule=True)

    direct: dict[int, tuple[str, str, dict[str, Any]]] = {
        16: ("current_time", "current local system time", {}),
        17: ("environment_summary", "system identity", {}),
        18: ("cpu_info", "CPU identity and core counts", {}),
        19: ("host_snapshot", "host resource state", {}),
        20: ("ollama_runtime_snapshot", "Ollama runtime state", {}),
        21: ("dns_query", "DNS resolution for example.com", {"target": "example.com", "record_type": "A"}),
        22: ("tcp_connect", "TCP connectivity to example.com:443", {"target": "example.com", "port": 443}),
        23: ("http_probe", "HTTPS probe for example.com", {"target": "https://example.com"}),
        24: ("page_metadata", "example.com page metadata", {"target": "https://example.com"}),
        30: ("write_file", "create disposable target file", {
            "target": "generalized_recipe_test/targets.txt",
            "content": "example.com\nwww.iana.org\n",
        }),
        31: ("read_file", "read disposable target file", {"target": "generalized_recipe_test/targets.txt"}),
        33: ("search_recipes", "semantic search for generalized endpoint recipe", {
            "query": "public hostname endpoint health DNS TCP HTTPS page metadata",
        }),
        34: ("load_recipe", "inspect existing generalized endpoint recipe", {"conditional": "existing_equivalent"}),
        36: ("save_recipe", "create parameterized public endpoint recipe", {"conditional": "no_equivalent"}),
        37: ("search_recipes", "post-create generalized recipe discovery", {"phase": "post_create"}),
        38: ("load_recipe", "inspect parameterized stored recipe", {"phase": "parameterization"}),
        39: ("run_recipe", "execute generalized recipe for example.com", {"hostname": "example.com", "phase": "first_replay"}),
        43: ("run_recipe", "execute generalized recipe for www.iana.org", {"hostname": "www.iana.org", "phase": "second_replay"}),
        47: ("load_recipe", "inspect generalized recipe after replay", {"phase": "post_replay"}),
        50: ("search_recipes", "semantic rediscovery using alternate wording", {
            "query": "check whether a website host resolves accepts TLS web connections responds over HTTPS identify its page",
            "phase": "alternate_query",
        }),
        59: ("remove_path", "remove disposable generalized recipe test directory", {"target": "generalized_recipe_test", "recursive": True}),
        60: ("path_stat", "verify disposable directory was removed", {"target": "generalized_recipe_test"}),
        61: ("search_recipes", "verify reusable recipe remains after cleanup", {"phase": "post_cleanup"}),
    }
    if number in direct:
        tool, label, extra = direct[number]
        return req(tool, label, **extra)

    labels = {
        25: "network/content layer audit", 26: "observation capability discovery",
        27: "skill capability discovery", 28: "recipe capability discovery",
        29: "capability provenance audit", 32: "workspace boundary audit",
        35: "recipe creation branch selection", 40: "first replay target substitution audit",
        41: "first replay direct-evidence comparison", 42: "first replay completion audit",
        44: "second replay target substitution audit", 45: "second replay target evidence audit",
        46: "two-replay parameterization comparison", 48: "stored recipe result-capture audit",
        49: "second-target duplicate recipe audit", 51: "direct primitive routing audit",
        52: "tool_search necessity audit", 53: "tool_search provenance persistence audit",
        54: "direct versus derived evidence audit", 55: "actual observation truncation audit",
        56: "recursive read_observation audit", 57: "retry/fallback audit",
        58: "first replay evidence preservation audit", 62: "cleanup mutation-boundary audit",
        63: "complete requirement ledger audit", 64: "requirements beyond 24 persistence audit",
        65: "terminal requirement field preservation audit", 66: "discovery provenance serialization audit",
        67: "bounded prompt-render audit", 68: "PASS evidence audit",
        69: "terminal-state exclusivity audit", 70: "pending-requirement audit",
        71: "single equivalent recipe audit", 72: "deterministic finalization audit",
    }
    return req(f"__genrecipe_audit_{number:02d}__", labels.get(number, f"generalized recipe audit {number}"), derived=True)


def derive_generalized_recipe_stress_requirements(user_text: str) -> list[Requirement]:
    items = _generalized_recipe_numbered_items(user_text)
    if not items:
        return []
    return [_generalized_recipe_requirement(number, section, body) for number, section, body in items]


def _tool_recipe_numbered_items(user_text: str) -> list[tuple[int, str, str]]:
    """Parse the numbered requirements from the tool/recipe stress prompt.

    The explicit A-L capability headings form the executable plan boundary, so
    the numbered GENERAL SAFETY RULES preamble is never misclassified as work.
    """
    text = str(user_text or "")
    if "tool-routing and recipe-system stress test" not in text.lower():
        return []
    headings = list(_TOOL_RECIPE_STRESS_SECTION_RE.finditer(text))
    if not headings:
        return []
    items: list[tuple[int, str, str]] = []
    for idx, heading in enumerate(headings):
        section = f"{heading.group(1).upper()}. {heading.group(2).strip()}"
        start = heading.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        final_output = re.search(r"(?mi)^\s*FINAL OUTPUT\s*$", text[start:end])
        if final_output:
            end = start + final_output.start()
        block = text[start:end]
        matches = list(_NUMBERED_LINE_RE.finditer(block))
        for j, match in enumerate(matches):
            item_start = match.end()
            item_end = matches[j + 1].start() if j + 1 < len(matches) else len(block)
            body = block[item_start:item_end].strip()
            if body:
                items.append((int(match.group(1)), section, body))
    seen: set[int] = set()
    ordered: list[tuple[int, str, str]] = []
    for row in sorted(items, key=lambda value: value[0]):
        if row[0] in seen:
            continue
        seen.add(row[0])
        ordered.append(row)
    return ordered


def _tool_recipe_requirement(number: int, section: str, body: str) -> Requirement:
    """Compile one tool/recipe stress item into a deterministic requirement."""
    lower = " ".join(body.lower().split())
    scope: dict[str, Any] = {
        "item_number": int(number),
        "section": section,
        "source_text": body[:2400],
        "tool_recipe_stress": True,
    }
    key = f"tooltest:{number:02d}"

    def req(tool: str, label: str, *, derived: bool = False, **extra: Any) -> Requirement:
        scope.update({k: v for k, v in extra.items() if v not in (None, "", [])})
        if derived:
            scope["derived"] = True
        return Requirement(key=key, tool=tool, label=label, scope=dict(scope))

    direct: dict[int, tuple[str, str, dict[str, Any]]] = {
        1: ("current_time", "current local/system time", {"fact_type": "current_time", "time_scope": "current"}),
        2: ("environment_summary", "basic host/system identity", {}),
        3: ("cpu_info", "CPU identity and core counts", {}),
        4: ("host_snapshot", "host resource state", {}),
        5: ("temperature_sensors", "temperature sensor state", {}),
        6: ("ollama_runtime_snapshot", "Ollama runtime state", {}),
        7: ("dns_query", "DNS resolution for example.com", {"target": "example.com", "record_type": "A"}),
        8: ("tcp_connect", "TCP connectivity to example.com:443", {"target": "example.com", "port": 443}),
        9: ("http_probe", "HTTPS probe for example.com", {"target": "https://example.com"}),
        10: ("page_metadata", "example.com page metadata", {"target": "https://example.com"}),
        15: ("write_file", "create disposable workspace test file", {
            "target": "harness_tool_recipe_test/input.txt",
            "content": "TOOL_PATH_TEST_OK\nalpha\nbeta\ngamma\n",
        }),
        16: ("read_file", "read disposable workspace test file", {"target": "harness_tool_recipe_test/input.txt"}),
        18: ("search_recipes", "semantic search for equivalent recipe", {"query": "quick local agent health check current time host resources cpu ollama"}),
        19: ("load_recipe", "existing recipe inspection/reuse branch", {"conditional": "existing_equivalent"}),
        20: ("save_recipe", "create one reusable health-check recipe when needed", {"conditional": "no_equivalent"}),
        21: ("search_recipes", "post-create recipe discovery and duplicate check", {"query": "quick local agent health check current time host resources cpu ollama", "phase": "post_create"}),
        22: ("run_recipe", "execute selected health-check recipe", {"recipe_name": "quick_local_agent_health_check"}),
        30: ("load_recipe", "stored recipe procedural-integrity inspection", {"recipe_name": "quick_local_agent_health_check", "phase": "integrity"}),
        33: ("remove_path", "remove disposable workspace test directory", {"target": "harness_tool_recipe_test", "recursive": True}),
    }
    if number in direct:
        tool, label, extra = direct[number]
        return req(tool, label, **extra)

    derived_labels = {
        11: ("__tooltest_layer_consistency__", "network/content layer consistency"),
        12: ("__tooltest_observation_capability__", "observation retrieval capability discovery"),
        13: ("__tooltest_skill_capability__", "skill discovery capability discovery"),
        14: ("__tooltest_recipe_capabilities__", "recipe capability discovery"),
        17: ("__tooltest_workspace_boundary__", "workspace path-boundary verification"),
        23: ("__tooltest_recipe_compare__", "compare recipe output with direct evidence"),
        24: ("__tooltest_recipe_replay_audit__", "recipe replay verification"),
        25: ("__tooltest_direct_routing_audit__", "direct primitive routing audit"),
        26: ("__tooltest_tool_search_audit__", "tool_search necessity audit"),
        27: ("__tooltest_retry_audit__", "retry/fallback audit"),
        28: ("__tooltest_truncation_audit__", "actual observation truncation audit"),
        29: ("__tooltest_recursive_truncation_audit__", "recursive read_observation truncation audit"),
        31: ("__tooltest_recipe_secret_audit__", "stored recipe secret/transient-data audit"),
        32: ("__tooltest_recipe_duplicate_audit__", "duplicate recipe audit"),
        34: ("__tooltest_mutation_audit__", "mutation boundary audit"),
        35: ("__tooltest_evidence_audit__", "successful-requirement evidence audit"),
        36: ("__tooltest_terminal_state_audit__", "terminal requirement-state audit"),
        37: ("__tooltest_deterministic_finalization__", "deterministic finalization audit"),
    }
    tool, label = derived_labels.get(number, (f"__unmapped_tooltest_item_{number:02d}__", f"tool/recipe test item {number}"))
    return req(tool, label, derived=True)


def derive_tool_recipe_stress_requirements(user_text: str) -> list[Requirement]:
    items = _tool_recipe_numbered_items(user_text)
    if not items:
        return []
    return [_tool_recipe_requirement(number, section, body) for number, section, body in items]


def _stress_numbered_items(user_text: str) -> list[tuple[int, str, str]]:
    """Return numbered requirements only from explicit capability-test sections.

    Long stress prompts commonly contain a numbered EXECUTION RULES preamble.
    Treating those numbers as tasks is as harmful as dropping the real tasks, so
    only the named capability sections participate in this compiler.
    """
    text = str(user_text or "")
    headings = list(_STRESS_SECTION_RE.finditer(text))
    if not headings:
        return []
    items: list[tuple[int, str, str]] = []
    for idx, heading in enumerate(headings):
        section = heading.group(1).strip()
        start = heading.end()
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        final_output = re.search(r"(?mi)^\s*FINAL OUTPUT\s*$", text[start:end])
        if final_output:
            end = start + final_output.start()
        block = text[start:end]
        matches = list(_NUMBERED_LINE_RE.finditer(block))
        for j, match in enumerate(matches):
            item_start = match.end()
            item_end = matches[j + 1].start() if j + 1 < len(matches) else len(block)
            body = block[item_start:item_end].strip()
            if body:
                items.append((int(match.group(1)), section, body))
    # Preserve source order and protect against accidentally parsing repeated
    # examples in a malformed prompt.
    seen: set[int] = set()
    ordered: list[tuple[int, str, str]] = []
    for row in sorted(items, key=lambda value: value[0]):
        if row[0] in seen:
            continue
        seen.add(row[0]); ordered.append(row)
    return ordered


def _stress_requirement(number: int, section: str, body: str) -> Requirement | None:
    """Compile one explicit stress-test item into a deterministic requirement."""
    lower = " ".join(body.lower().split())
    scope: dict[str, Any] = {
        "item_number": int(number),
        "section": section,
        "source_text": body[:2000],
    }
    key = f"stress:{number:02d}"

    def req(tool: str, label: str, **extra: Any) -> Requirement:
        scope.update({k: v for k, v in extra.items() if v not in (None, "", [])})
        return Requirement(key=key, tool=tool, label=label, scope=dict(scope))

    # Cross-capability items are harness-derived checks over already collected
    # evidence. Resolve them before ordinary lexical rules so phrases such as
    # "HTTPS probe" in a consistency check do not create a duplicate network call.
    if number == 21 or "compare the system time" in lower:
        return req("__derived_time_consistency__", "system/remote time consistency", derived=True)
    if number == 22 or ("page retrieval result for example.com" in lower and "https probe" in lower):
        return req("__derived_reachability_consistency__", "example.com reachability consistency", derived=True)
    if number == 23 or "every successful requirement has actual tool evidence" in lower:
        return req("__derived_evidence_audit__", "successful-requirement evidence audit", derived=True)
    if number == 24 or "truncation warnings" in lower or "middle was truncated" in lower:
        return req("__derived_truncation_audit__", "truncation/retrieval audit", derived=True)

    if re.search(r"\bcurrent weather\b", lower):
        return req("weather_forecast", "current weather for London, Ontario", fact_type="weather", entity="London, Ontario", time_scope="current")
    if "latest" in lower and re.search(r"\b(?:news|headlines?)\b", lower):
        return req("news_search", "latest 3 local London, Ontario headlines", fact_type="news", entity="London, Ontario, Canada", time_scope="latest", limit=3)
    if "brent" in lower and re.search(r"\b(?:price|quote)\b", lower):
        return req("market_quote", "current Brent crude oil price", fact_type="market_price", instruments=["brent"], time_scope="current")
    if number == 4 and "example.com" in lower:
        return req("page_metadata", "example.com remote page retrieval", target="https://example.com")
    if re.search(r"\bcurrent (?:system(?:/local)?|local) time\b", lower):
        return req("current_time", "current system/local time", fact_type="current_time", time_scope="current")
    if "operating-system information" in lower or "kernel/system name" in lower:
        return req("environment_summary", "operating-system/kernel identity")
    if "harmless shell" in lower or "harness_system_tool_ok" in lower:
        return req("execute_shell", "read-only shell execution marker", command="printf 'HARNESS_SYSTEM_TOOL_OK\\n'")
    if "filesystem" in lower and re.search(r"\bfree space\b", lower):
        return req("filesystem_snapshot", "workspace filesystem free-space state")
    if "host snapshot" in lower or ("uptime" in lower and "total memory" in lower):
        return req("host_snapshot", "host resource snapshot")
    if "cpu identity" in lower or "logical cpu count" in lower:
        return req("cpu_info", "CPU identity and logical CPU count")
    if "thermal" in lower or "temperature sensors" in lower:
        return req("temperature_sensors", "host thermal sensors")
    if "ollama" in lower and "running" in lower:
        return req("ollama_runtime_snapshot", "Ollama runtime status")
    if re.search(r"^gmail\s*:", lower):
        return req("gmail_search_messages", "Gmail read-only inbox summary", query="in:inbox", limit=3)
    if "google calendar" in lower:
        return req("google_calendar_list_events", "next 3 Google Calendar events", limit=3)
    if "google drive" in lower:
        # The capability may not be installed. Keeping the explicit tool name in
        # the ledger lets the schema gate mark it unresolved instead of dropping it.
        return req("google_drive_list_files", "3 most recently modified Google Drive files", limit=3)
    if number == 16 and "example.com" in lower and "resolve" in lower:
        return req("dns_query", "DNS resolution for example.com", target="example.com", record_type="A")
    if "tcp connectivity" in lower and "example.com" in lower:
        return req("tcp_connect", "TCP connectivity to example.com:443", target="example.com", port=443)
    if "https probe" in lower and "example.com" in lower:
        return req("http_probe", "HTTPS probe for example.com", target="https://example.com")
    if "harness-stress-test-invalid.example" in lower:
        return req("dns_query", "expected DNS failure for invalid hostname", target="harness-stress-test-invalid.example", record_type="A", expect_dns_failure=True)
    if "127.0.0.1:11434/api/version" in lower:
        return req("http_probe", "Ollama API reachability", target="http://127.0.0.1:11434/api/version", allow_private=True)
    return None


def derive_stress_requirements(user_text: str) -> list[Requirement]:
    items = _stress_numbered_items(user_text)
    if not items:
        return []
    result: list[Requirement] = []
    for number, section, body in items:
        requirement = _stress_requirement(number, section, body)
        if requirement is not None:
            result.append(requirement)
        else:
            # Never silently discard a numbered stress-test requirement. Unknown
            # items are represented explicitly and will be closed as unavailable.
            result.append(Requirement(
                key=f"stress:{number:02d}",
                tool=f"__unmapped_stress_item_{number:02d}__",
                label=f"stress-test item {number}",
                scope={"item_number": number, "section": section, "source_text": body[:2000]},
            ))
    return result

def _scope_for_requirement(key: str, tool: str, text: str, frame: dict[str, Any]) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    fact_intent = _FACT_RULE_INTENTS.get(key) or _TOOL_FACT_INTENTS.get(tool)
    if fact_intent:
        scope["fact_type"] = fact_intent
    target = _extract_target(tool, text)
    if target:
        scope["target"] = target
    if key.startswith("weather_") or frame.get("intent") == "weather" and tool in {"web_search", "browse_url"}:
        if frame.get("entity"):
            scope["entity"] = frame["entity"]
        if frame.get("time_scope"):
            scope["time_scope"] = frame["time_scope"]
        scope["fact_type"] = "weather"
    if frame.get("intent") == "news" and tool == "news_search":
        if frame.get("entity"):
            scope["entity"] = frame["entity"]
        if frame.get("time_scope"):
            scope["time_scope"] = frame["time_scope"]
        scope["fact_type"] = "news"
    if frame.get("intent") == "market_price" and tool == "market_quote":
        if frame.get("instruments"):
            scope["instruments"] = list(frame["instruments"])
        scope["fact_type"] = "market_price"
    return scope


def _scope_text(arguments: Any, result_text: str = "") -> str:
    try:
        import json
        args = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        args = str(arguments or "")
    return (args + " " + str(result_text or "")).lower()


def _scope_matches(
    scope: dict[str, Any], arguments: Any, result_text: str = "", result_metadata: dict[str, Any] | None = None
) -> bool:
    if not scope:
        return True
    # Scope metadata is a strengthening of the runtime contract.  Older callers
    # and persisted observations may not provide arguments/result text; treat
    # missing provenance as unknown rather than as a mismatched target.  New
    # execution paths always pass both, so wrong-target successes are rejected.
    if arguments in (None, {}, "") and not str(result_text or "").strip() and not result_metadata:
        return True
    haystack = _scope_text(arguments, result_text)
    target = str(scope.get("target") or "").lower().rstrip(".")
    if target and target not in haystack:
        return False
    entity = str(scope.get("entity") or "").strip().lower()
    if entity:
        tokens = [t for t in re.findall(r"[a-z0-9]+", entity) if len(t) > 2 and t not in {"the", "city", "region"}]
        if tokens and not all(token in haystack for token in tokens):
            return False
    # Time scope is intentionally not required for browse_url: a verified page can
    # inherit the time scope from the linked search observation. Grounding performs
    # the stricter search->browse provenance check.
    instruments = [str(item).lower() for item in (scope.get("instruments") or []) if str(item)]
    if instruments:
        expected = set(extract_market_instruments(" ".join(instruments)) or instruments)
        returned: set[str] = set()
        metadata = dict(result_metadata or {})
        if metadata.get("market_instruments") is not None:
            returned.update(extract_market_instruments(" ".join(str(item) for item in (metadata.get("market_instruments") or []))))
            if not expected.issubset(returned):
                return False
        elif str(result_text or "").strip():
            try:
                payload = json.loads(str(result_text))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("quotes"), list):
                for row in payload.get("quotes") or []:
                    if not isinstance(row, dict) or not isinstance(row.get("price"), (int, float)):
                        continue
                    for value in (row.get("instrument"), row.get("symbol")):
                        returned.update(extract_market_instruments(str(value or "")))
                # When provider rows are available they, not requested call args,
                # prove coverage. This prevents a partial multi-asset response
                # from prematurely satisfying the requirement ledger.
                if not expected.issubset(returned):
                    return False
            else:
                returned = set()
        if not returned and (str(result_text or "").strip() or metadata):
            # Modern execution supplied result provenance but it did not prove any
            # numeric quote rows. Never fall back to requested call arguments: a
            # malformed/empty provider response cannot satisfy market scope.
            return False
        if not returned and not str(result_text or "").strip() and not metadata:
            # Legacy caller fallback when no result provenance was supplied.
            actual = []
            if isinstance(arguments, dict):
                actual = extract_market_instruments(" ".join(str(item) for item in (arguments.get("instruments") or [])))
            if actual and not expected.issubset(set(actual)):
                return False
    if scope.get("time_scope") and isinstance(arguments, dict) and "query" in arguments:
        expected = str(scope["time_scope"]).lower()
        temporal_tokens = [t for t in re.findall(r"[a-z0-9]+", expected) if len(t) > 2]
        if scope.get("fact_type") == "news" and expected in {"current", "latest", "recent"}:
            # The deterministic news query builder canonicalizes an unspecified
            # current scope to ``latest``. Treat the ordinary current-news words
            # as equivalent so a correctly scoped search is not left pending
            # simply because the request said "current" and the query said
            # "latest".
            temporal_tokens = ["current", "latest", "recent", "today"]
        if temporal_tokens and not any(token in haystack for token in temporal_tokens):
            return False
    return True


def derive_requirements(user_text: str) -> list[Requirement]:
    """Return ordered, deduplicated requirements explicitly present in a request."""
    text = str(user_text or "")
    generalized_recipe_stress = derive_generalized_recipe_stress_requirements(text)
    if generalized_recipe_stress:
        return generalized_recipe_stress
    tool_recipe_stress = derive_tool_recipe_stress_requirements(text)
    if tool_recipe_stress:
        return tool_recipe_stress
    stress = derive_stress_requirements(text)
    if stress:
        return stress
    lower = text.lower().replace("_", " ")
    result: list[Requirement] = []
    seen_tools: set[str] = set()
    fact_frames = derive_fact_frames(text)
    frame = select_primary_fact_frame(fact_frames, derive_task_frame(text))

    for key, tool, label, patterns in _RULES:
        expected_intent = _FACT_RULE_INTENTS.get(key)
        scoped_frame = dict(fact_frames.get(expected_intent) or frame) if expected_intent else frame
        if key == "weather_forecast" and is_evidence_reuse_request(text):
            continue
        if expected_intent and expected_intent not in fact_frames and is_implementation_request(text):
            continue
        if any(re.search(pattern, lower, flags=re.I) for pattern in patterns):
            if tool not in seen_tools:
                result.append(Requirement(key=key, tool=tool, label=label, scope=_scope_for_requirement(key, tool, text, scoped_frame)))
                seen_tools.add(tool)

    raw_lower = text.lower()
    for tool in sorted(_EXPLICIT_TOOL_NAMES):
        if tool in raw_lower and tool not in seen_tools:
            fact_intent = _TOOL_FACT_INTENTS.get(tool, "")
            scoped_frame = dict(fact_frames.get(fact_intent) or frame) if fact_intent else frame
            result.append(Requirement(key=f"explicit:{tool}", tool=tool, label=f"explicitly requested {tool}", scope=_scope_for_requirement(f"explicit:{tool}", tool, text, scoped_frame)))
            seen_tools.add(tool)
    return result


def is_evidence_reuse_request(user_text: str) -> bool:
    """Detect short commands that ask to present already-collected evidence.

    These should reuse the preceding working-state observation rather than start
    a fresh web/system lookup merely because a noun such as ``forecast`` is
    present.
    """
    text = " ".join(str(user_text or "").strip().split())
    if not text or len(text) > 180:
        return False
    lower = text.lower()
    if not re.match(r"^(?:display|show|summarize|summarise|repeat|list|give me|present)\b", lower):
        return False
    return bool(re.search(r"\b(?:forecast|results?|findings?|report|answer|data|output|that|those|previous|above)\b", lower))


_META_CAPABILITY_REQUEST_RE = re.compile(
    r"^(?:what\s+else\s+)?(?:can|could|would)\s+you\s+(?:do|help\s+with|handle|support)\b"
    r"|^(?:what|which)\s+(?:tools|capabilities|features|things)\s+(?:can|do)\s+you\b"
    r"|^what\s+are\s+you\s+capable\s+of\b",
    re.I,
)


def is_meta_capability_request(user_text: str) -> bool:
    """Return True for self-contained capability/help questions.

    These often begin with phrases such as "what else" but are not referential
    continuations of the prior task. Treating them as continuations can inherit a
    stale weather/news/task frame and trigger unrelated tools.
    """
    text = " ".join(str(user_text or "").strip().split())
    return bool(text and _META_CAPABILITY_REQUEST_RE.search(text))


def is_followup_request(user_text: str) -> bool:
    """Conservatively detect requests that intentionally depend on the prior task.

    Long, self-contained requests start a fresh task epoch.  Only explicit
    continuity language or short referential follow-ups carry prior task evidence.
    """
    text = " ".join(str(user_text or "").strip().split())
    if not text:
        return False
    if is_meta_capability_request(text):
        return False
    lower = text.lower()
    continuity = (
        "without rerunning", "without re-running", "continue", "continue from", "based on that",
        "based on the previous", "previous result", "previous task", "earlier result", "same task",
        "what remains", "what did you find", "those results", "these results", "that result",
        "the above", "follow up", "follow-up",
    )
    # Referential words inside a long, self-contained specification are not a
    # continuation signal. For example, a safety rule saying "before summarizing
    # that result" used to collapse the requirement ledger to the previous/primary
    # fact frame. Long follow-ups still work when the continuity cue is actually
    # leading the request ("Based on that result, ...").
    leading = lower[:240].lstrip()
    if any(phrase in leading for phrase in continuity):
        return True
    if len(text) <= 240 and any(phrase in lower for phrase in continuity):
        return True
    if is_evidence_reuse_request(text):
        return True
    if len(text) <= 180 and re.match(r"^(?:what about|what else)\b", lower):
        return True
    if len(text) <= 80 and re.match(r"^(?:and|also)\b", lower):
        # Fragments like "And Toronto?" are referential; imperative continuations
        # such as "And write a Python script" start a new task.
        return not bool(re.search(r"\b(?:write|create|make|run|execute|build|generate|explain|summarize|show|tell)\b", lower))
    return False


@dataclass
class TaskRequirementLedger:
    requirements: list[Requirement] = field(default_factory=list)

    @classmethod
    def from_request(cls, user_text: str) -> "TaskRequirementLedger":
        return cls(derive_requirements(user_text))

    def required_tools(self, pending_only: bool = False) -> list[str]:
        rows = [item for item in self.requirements if not bool((item.scope or {}).get("derived"))]
        if pending_only:
            rows = [item for item in rows if item.status not in {"satisfied", "partial", "blocked"}]
        # Tool schemas are unique even when several independent requirements use
        # the same primitive with different scopes (for example two DNS probes).
        return list(dict.fromkeys(item.tool for item in rows))

    def record_tool(
        self, tool_name: str, *, status: str, reason: str = "", fingerprint: str = "",
        arguments: Any = None, result_text: str = "", result_metadata: dict[str, Any] | None = None,
    ) -> None:
        tool_name = str(tool_name or "")
        targets = {tool_name}
        if status in {"ok", "partial"}:
            targets.update(_EQUIVALENT_REQUIREMENT_TOOLS.get(tool_name, ()))
        direct_rows = [item for item in self.requirements if item.tool == tool_name]
        direct_match_exists = any(
            _scope_matches(item.scope, arguments, result_text, result_metadata) for item in direct_rows
        )
        for item in self.requirements:
            if item.tool not in targets:
                continue
            direct = item.tool == tool_name
            scoped = _scope_matches(item.scope, arguments, result_text, result_metadata)
            # Several independent requirements may intentionally use the same
            # primitive with different targets (e.g. example.com and an expected
            # NXDOMAIN DNS probe). A call matching one must not count as a failed
            # attempt against its siblings. Preserve the older wrong-target
            # diagnostic only when no same-tool requirement matches the call.
            if direct and not scoped and direct_match_exists:
                continue
            if direct:
                item.attempts += 1
            item.last_reason = str(reason or "")[:120]
            item.fingerprint = str(fingerprint or "")[:32]
            if status == "ok" and scoped:
                item.status = "satisfied"
            elif status == "partial" and scoped:
                item.status = "partial"
            elif direct and status in {"ok", "partial"} and not scoped:
                item.status = "failed"
                item.last_reason = "successful tool result did not match requested target/scope"
            elif direct and item.status not in {"satisfied", "partial"}:
                item.status = "failed"

    def record_tool_for_key(
        self, key: str, tool_name: str, *, status: str, reason: str = "", fingerprint: str = "",
        arguments: Any = None, result_text: str = "", result_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a deterministic tool result against exactly one requirement.

        Sectioned stress plans may intentionally call the same primitive more
        than once with identical arguments at different workflow phases (for
        example recipe search before and after creation).  Global tool matching
        would prematurely satisfy the later phase, so harness-owned execution
        uses this key-scoped form.
        """
        for item in self.requirements:
            if item.key != str(key):
                continue
            if item.tool != str(tool_name):
                item.status = "failed"
                item.last_reason = "deterministic executor used an unexpected tool"
                return
            item.attempts += 1
            item.last_reason = str(reason or "")[:120]
            item.fingerprint = str(fingerprint or "")[:32]
            scoped = _scope_matches(item.scope, arguments, result_text, result_metadata)
            if status == "ok" and scoped:
                item.status = "satisfied"
            elif status == "partial" and scoped:
                item.status = "partial"
            elif status in {"ok", "partial"} and not scoped:
                item.status = "failed"
                item.last_reason = "successful tool result did not match requested target/scope"
            else:
                item.status = "failed"
            return

    def record_evidence_for_key(
        self, key: str, *, source: str, tool_name: str = "", status: str = "",
        reason: str = "", fingerprint: str = "", arguments_digest: str = "",
        evidence_ref: str = "", count_attempt: bool = False,
    ) -> None:
        """Attach explicit provenance to one requirement without changing its outcome.

        Derived requirements can be satisfied by harness-owned inspections rather
        than by their pseudo-tool name.  Recording that provenance separately keeps
        the requirement auditable without pretending the discovery tool itself was
        the requirement's execution primitive.
        """
        for item in self.requirements:
            if item.key != str(key):
                continue
            if count_attempt:
                item.attempts += 1
            row = {
                "source": str(source or "")[:32],
                "tool": str(tool_name or "")[:80],
                "status": str(status or "")[:24],
                "reason": str(reason or "")[:120],
                "fingerprint": str(fingerprint or "")[:32],
                "arguments_digest": str(arguments_digest or "")[:24],
                "evidence_ref": str(evidence_ref or "")[:80],
            }
            row = {k: v for k, v in row.items() if v not in (None, "")}
            if row and row not in item.evidence:
                item.evidence.append(row)
                item.evidence[:] = item.evidence[-4:]
            return

    def mark_fact_satisfied(self, fact_type: str, reason: str = "grounding_evidence") -> None:
        """Close tool requirements whose requested fact was grounded by any valid path."""
        fact_type = str(fact_type or "")
        for item in self.requirements:
            if str((item.scope or {}).get("fact_type") or "") == fact_type:
                item.status = "satisfied"
                item.last_reason = str(reason or "")[:120]

    def status_for_tool(self, tool_name: str) -> str:
        for item in self.requirements:
            if item.tool == tool_name:
                return item.status
        return ""

    def completed_tools(self) -> set[str]:
        return {item.tool for item in self.requirements if item.status in {"satisfied", "partial"}}

    def closed_tools(self) -> set[str]:
        return {item.tool for item in self.requirements if item.status in {"satisfied", "partial", "blocked"}}

    def pending(self) -> list[Requirement]:
        return [item for item in self.requirements if item.status not in {"satisfied", "partial", "blocked"}]

    def mark_blocked(self, tool_name: str, reason: str) -> None:
        for item in self.requirements:
            if item.tool == tool_name and item.status not in {"satisfied", "partial"}:
                item.status = "blocked"
                item.last_reason = str(reason or "")[:120]

    def mark_key(self, key: str, status: str, reason: str = "") -> None:
        """Set one requirement by key, including derived/non-tool checks."""
        for item in self.requirements:
            if item.key != key:
                continue
            item.status = str(status or "pending")
            item.last_reason = str(reason or "")[:120]
            return

    def block_exhausted(self, limit: int, reason: str = "per-requirement retry budget exhausted") -> list[str]:
        """Freeze failed requirements after their bounded direct-attempt budget."""
        limit = max(1, int(limit))
        blocked: list[str] = []
        for item in self.requirements:
            if item.status == "failed" and item.attempts >= limit:
                item.status = "blocked"
                item.last_reason = str(reason or "")[:120]
                blocked.append(item.tool)
        return blocked

    def as_list(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self.requirements]

    def pending_hint(self, limit: int = 8) -> str:
        pending = self.pending()
        if not pending:
            return ""
        limit = max(1, min(int(limit), 20))
        rows = ", ".join(f"{item.tool}[{item.status}]" for item in pending[:limit])
        more = f", +{len(pending) - limit} more" if len(pending) > limit else ""
        return (
            "[Harness pending requirements] Do not draft the final report yet. "
            f"Complete an unfinished explicit check with a supplied native tool. Pending: {rows}{more}. "
            "Reuse satisfied evidence; do not repeat completed checks. If validator recovery guidance is present, follow it first."
        )

    def completion_message(self) -> str:
        pending = self.pending()
        if not pending:
            return ""
        rows = [f"- {item.label} -> {item.tool} (status={item.status}, attempts={item.attempts})" for item in pending]
        return (
            "[Harness completion gate] The current request explicitly requires checks that have not yet been completed. "
            "Do not provide a final answer yet and do not claim these capabilities are unavailable unless the corresponding "
            "tool has actually failed or is blocked by harness policy. Complete the missing checks using the supplied schemas:\n"
            + "\n".join(rows[:20])
        )
