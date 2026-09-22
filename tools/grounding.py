"""Deterministic fact-type grounding gates for final-answer control.

The main/fast models may decide *how* to recover, but they cannot declare a
fact-retrieval turn grounded.  This module derives a small set of requested fact
types from the user request and checks trusted tool provenance plus bounded
content signals before finalization.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .task_requirements import classify_request_intent, derive_fact_frames, derive_task_frame, is_implementation_request
from .market import extract_market_instruments, is_market_price_request

WEATHER_RECIPE_NAME = "weather.current_forecast"

_WEATHER_REQUEST_PATTERNS = (
    re.compile(r"\bwhat(?:'s| is) (?:the )?(?:weather|forecast)\b", re.I),
    re.compile(r"\b(?:weather|forecast|temperature|precipitation|rain|snow|humidity|wind)\b.*\b(?:today|tomorrow|current|now|tonight|week|days?|hours?)\b", re.I),
    re.compile(r"\b(?:find|check|show|get|give|tell me|look up)\b.*\b(?:weather|forecast|temperature|precipitation|rain|snow|humidity|wind)\b", re.I),
)
_TIME_REQUEST_RE = re.compile(
    r"\b(what time is it|current time|current date|today(?:'s)? date|what day is it|local time|utc time|timezone)\b",
    re.I,
)
_HOST_REQUEST_RE = re.compile(
    r"\b(?:host (?:health|cpu|memory|disk|temperature|state)|cpu.*memory|memory.*disk|system health|host snapshot)\b",
    re.I,
)
_NETWORK_REQUEST_RE = re.compile(
    r"\b(?:network (?:interfaces|routes|health|state|status|connections?)|listening sockets?|active connections?|neighbor table|arp|ndp)\b",
    re.I,
)
_REPO_REQUEST_RE = re.compile(
    r"\b(?:repo(?:sitory)? (?:status|diff|health)|git status|git diff|repository checks?)\b",
    re.I,
)
_WEB_FACT_REQUEST_RE = re.compile(
    r"\b(?:search the web|look up|web research|verify .*source|current .*documentation|official .*documentation)\b",
    re.I,
)
_NEWS_REQUEST_RE = re.compile(
    r"(?:\b(?:latest|recent|current|today(?:'s)?)\b.{0,48}\b(?:news|headlines?|stories?)\b"
    r"|\b(?:news|headlines?)\b.{0,48}\b(?:latest|recent|current|today)\b"
    r"|\b(?:latest|top|local)\s+headlines?\b)",
    re.I,
)
_ENCYCLOPEDIC_REQUEST_RE = re.compile(
    r"^\s*(what|who)\s+(?:is|are|was|were)\s+(?:(a|an|the)\s+)?(.{1,180}?)\s*[?!.]*$",
    re.I,
)
_ENCYCLOPEDIC_EXCLUDE_RE = re.compile(
    r"\b(?:your|my|our|today|tomorrow|current|latest|weather|forecast|time|date|timezone|"
    r"temperature|price|cost|score|result|status|version|ip address|hostname)\b",
    re.I,
)
_WEATHER_PRIMARY_RE = re.compile(r"\b(weather|forecast|current conditions?)\b", re.I)
_WEATHER_DETAIL_RE = re.compile(
    r"(?:\btemperature\b|\btemp\b|\bhighs?\b|\blows?\b|\bhumidity\b|\bwind(?:s|y)?\b|"
    r"\bprecipitation\b|\brain(?:fall|y)?\b|\bsnow(?:fall|y)?\b|\bfeels like\b|\bdew point\b|"
    r"\bchance of\b|\bvisibility\b|\bpressure\b|°\s*[CF]\b)",
    re.I,
)


@dataclass
class FactRequirement:
    fact_type: str
    status: str = "pending"
    evidence: list[str] = field(default_factory=list)
    last_error: str = ""

    @property
    def satisfied(self) -> bool:
        return self.status == "satisfied"

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_type": self.fact_type,
            "status": self.status,
            "satisfied": self.satisfied,
            "evidence": list(self.evidence),
            "last_error": self.last_error,
        }


@dataclass
class FactGroundingLedger:
    requirements: dict[str, FactRequirement] = field(default_factory=dict)

    @classmethod
    def from_fact_types(cls, fact_types: set[str] | list[str]) -> "FactGroundingLedger":
        return cls({str(name): FactRequirement(str(name)) for name in sorted(set(fact_types or []))})

    def apply_report(self, report: dict[str, Any]) -> None:
        evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
        missing = {str(item) for item in (report.get("missing_fact_types") or [])}
        reason = str(report.get("reason") or report.get("diagnosis") or "")[:240]
        for fact_type in report.get("required_fact_types") or []:
            key = str(fact_type)
            requirement = self.requirements.setdefault(key, FactRequirement(key))
            proof = [str(item) for item in (evidence.get(key) or []) if str(item)]
            if proof:
                requirement.status = "satisfied"
                requirement.evidence = proof
                requirement.last_error = ""
            elif key in missing and not requirement.satisfied:
                requirement.status = "pending"
                requirement.last_error = reason

    def mark_error(self, fact_type: str, reason: str) -> None:
        key = str(fact_type or "")
        requirement = self.requirements.setdefault(key, FactRequirement(key))
        if not requirement.satisfied:
            requirement.status = "failed"
            requirement.last_error = str(reason or "")[:240]

    def missing_fact_types(self) -> set[str]:
        return {key for key, value in self.requirements.items() if not value.satisfied}

    def can_finalize(self) -> bool:
        return all(item.satisfied for item in self.requirements.values())

    def as_list(self) -> list[dict[str, Any]]:
        return [self.requirements[key].as_dict() for key in sorted(self.requirements)]


def requested_fact_types(
    user_request: str,
    task_frame: dict[str, Any] | None = None,
    fact_frames: dict[str, dict[str, Any]] | None = None,
) -> set[str]:
    """Return fact types with deterministic grounding policies for this request."""
    text = " ".join(str(user_request or "").split())
    frame = dict(task_frame or {})
    result: set[str] = set()
    result.update(str(key) for key in dict(fact_frames or {}) if str(key))
    explicit_intent = classify_request_intent(text)
    implementation = is_implementation_request(text)
    if frame.get("intent") == "weather" or explicit_intent == "weather" or (not implementation and any(pattern.search(text) for pattern in _WEATHER_REQUEST_PATTERNS)):
        result.add("weather")
    if frame.get("intent") == "current_time" or explicit_intent == "current_time" or (not implementation and _TIME_REQUEST_RE.search(text)):
        result.add("current_time")
    if frame.get("intent") == "host_state" or explicit_intent == "host_state" or (not implementation and _HOST_REQUEST_RE.search(text)):
        result.add("host_state")
    if frame.get("intent") == "network_state" or explicit_intent == "network_state" or (not implementation and _NETWORK_REQUEST_RE.search(text)):
        result.add("network_state")
    if frame.get("intent") == "repository_state" or explicit_intent == "repository_state" or (not implementation and _REPO_REQUEST_RE.search(text)):
        result.add("repository_state")
    if frame.get("intent") == "news" or explicit_intent == "news" or (not implementation and _NEWS_REQUEST_RE.search(text)):
        result.add("news")
    if frame.get("intent") == "market_price" or explicit_intent == "market_price" or (not implementation and is_market_price_request(text)):
        result.add("market_price")
    if _WEB_FACT_REQUEST_RE.search(text) and not result and not implementation:
        result.add("web_fact")
    if not result and encyclopedic_lookup_query(text):
        result.add("encyclopedic")
    return result


def encyclopedic_lookup_query(user_request: str) -> str:
    """Return the bounded subject of a simple encyclopedic definition request.

    This deliberately excludes current/user-specific questions and arithmetic so
    stable definitions such as ``What is a shoggoth?`` can be grounded without
    turning philosophical questions such as ``Is God real?`` into fake fact
    retrieval tasks.
    """
    text = " ".join(str(user_request or "").strip().split())
    match = _ENCYCLOPEDIC_REQUEST_RE.match(text)
    if not match:
        return ""
    interrogative = str(match.group(1) or "").lower()
    article = str(match.group(2) or "").lower()
    subject = re.sub(r"\s+", " ", str(match.group(3) or "")).strip(" \t\r\n?!.;:")
    if not subject or len(subject) > 160 or len(subject.split()) > 12:
        return ""
    if _ENCYCLOPEDIC_EXCLUDE_RE.search(subject):
        return ""
    if re.search(r"\d\s*[+\-*/=]\s*\d", subject):
        return ""
    if subject.lower() in {"this", "that", "it", "there", "happening", "going on"}:
        return ""
    # Keep the automatic lookup narrow so conceptual/philosophical prompts such
    # as "What is love?" remain normal conversation.  Indefinite definitions
    # ("What is a shoggoth?"), named/acronym subjects, and "Who is ...?" are
    # the high-value cases where small-model fabricated specifics are most costly.
    if interrogative == "what" and article not in {"a", "an"}:
        first = subject.split()[0] if subject.split() else ""
        named = bool(first[:1].isupper()) or (len(first) >= 2 and first.isupper())
        if not named:
            return ""
    return subject

def _json_payload(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    # Harness status prefixes may precede JSON in some observation paths.
    candidates = [raw]
    start = min((idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0), default=-1)
    if start > 0:
        candidates.append(raw[start:])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
    return None


def _canonical_market_instruments(values: Any) -> set[str]:
    """Normalize persisted market names/symbols to the harness canonical names."""
    result: set[str] = set()
    for value in values or []:
        raw = str(value or "").strip()
        if not raw:
            continue
        canonical = extract_market_instruments(raw)
        if canonical:
            result.update(canonical)
        else:
            result.add(raw.lower())
    return result


def _market_quote_instruments(content: str) -> list[str]:
    """Return instruments that actually have numeric quotes in provider output.

    This is deliberately derived from the full tool result before working-state
    evidence is clipped.  A multi-instrument quote payload is commonly longer
    than the evidence preview budget, so reparsing the preview later is not a
    reliable proof that every requested instrument was returned.
    """
    payload = _json_payload(content)
    if not isinstance(payload, dict):
        return []
    found: set[str] = set()
    for row in payload.get("quotes") or []:
        if not isinstance(row, dict) or not isinstance(row.get("price"), (int, float)):
            continue
        values = [row.get("instrument"), row.get("symbol")]
        found.update(_canonical_market_instruments(values))
    return sorted(found)


def _content_terms(text: str, *, limit: int = 512) -> list[str]:
    """Return bounded exact lexical proof derived from the full observation.

    These terms are persisted for validator use only and are not rendered into
    the model prompt.  They let scope checks survive evidence-preview clipping
    without storing another copy of the raw page/tool output.
    """
    seen: set[str] = set()
    result: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", str(text or "").lower()):
        if len(token) <= 2 or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= max(64, int(limit)):
            break
    return result


def _geocode_candidates(content: str) -> list[dict[str, Any]]:
    payload = _json_payload(content)
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in payload[:10]:
        if not isinstance(item, dict):
            continue
        try:
            lat = float(item.get("latitude"))
            lon = float(item.get("longitude"))
        except (TypeError, ValueError):
            continue
        rows.append({
            "name": str(item.get("name") or "")[:160],
            "admin1": str(item.get("admin1") or "")[:120],
            "admin2": str(item.get("admin2") or "")[:120],
            "country": str(item.get("country") or "")[:120],
            "country_code": str(item.get("country_code") or "")[:12],
            "timezone": str(item.get("timezone") or "")[:80],
            "latitude": lat,
            "longitude": lon,
        })
    return rows


def _coordinates_from_weather(content: str, arguments: dict[str, Any]) -> dict[str, float]:
    payload = _json_payload(content)
    lat = payload.get("latitude") if isinstance(payload, dict) else None
    lon = payload.get("longitude") if isinstance(payload, dict) else None
    if lat is None:
        lat = arguments.get("latitude")
    if lon is None:
        lon = arguments.get("longitude")
    try:
        return {"latitude": float(lat), "longitude": float(lon)}
    except (TypeError, ValueError):
        return {}


def _time_proof(content: str) -> dict[str, Any]:
    payload = _json_payload(content)
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("timezone", "timezone_abbreviation", "utc_offset", "date", "time", "local", "utc"):
        value = payload.get(key)
        if value not in (None, ""):
            result[key] = str(value)[:160]
    if isinstance(payload.get("unix_timestamp"), int):
        result["unix_timestamp"] = int(payload["unix_timestamp"])
    return result


def _proof(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("grounding_proof")
    return dict(value) if isinstance(value, dict) else {}


def _coordinates_close(left: dict[str, Any], right: dict[str, Any], *, tolerance: float = 0.05) -> bool:
    try:
        return (
            abs(float(left.get("latitude")) - float(right.get("latitude"))) <= tolerance
            and abs(float(left.get("longitude")) - float(right.get("longitude"))) <= tolerance
        )
    except (TypeError, ValueError):
        return False


def is_weather_bearing(content: str) -> bool:
    """Conservatively detect whether an observation actually carries weather data."""
    text = str(content or "")
    if not text.strip():
        return False
    # Normalize common JSON/API field separators without trusting the source
    # hostname alone; an error page on a weather domain is not weather evidence.
    # Strip URLs before semantic inspection so ``https://weather.example/404``
    # does not become weather-bearing merely because of its hostname/path.
    probe = re.sub(r"https?://[^\s\"']+", " ", text, flags=re.I)
    probe = probe.replace("_", " ").replace("-", " ")
    primary = bool(_WEATHER_PRIMARY_RE.search(probe))
    details = {m.group(0).lower() for m in _WEATHER_DETAIL_RE.finditer(probe)}
    # A forecast/weather-labelled result is enough for search discovery. For
    # unlabeled API/text payloads require multiple meteorological fields.
    return primary or len(details) >= 2


def _recipe_stages(content: str) -> set[str]:
    payload = _json_payload(content)
    if not isinstance(payload, dict):
        return set()
    stages = payload.get("stages")
    if not isinstance(stages, list):
        return set()
    return {
        str(item.get("tool") or "")
        for item in stages
        if isinstance(item, dict)
        and item.get("ok") is True
        and not bool(item.get("skipped"))
    }


def classify_fact_types(tool_name: str, content: str) -> set[str]:
    """Classify fact types carried by one successful observation.

    Grounding-sensitive deterministic tools must satisfy their structured output
    contract. Arbitrary non-empty text is not evidence merely because it came
    from a tool whose name normally carries a fact type.
    """
    name = str(tool_name or "").strip().lower()
    text = str(content or "")
    payload = _json_payload(text)
    result: set[str] = set()
    if name == "current_time" and isinstance(payload, dict) and all(
        str(payload.get(key) or "").strip() for key in ("utc", "local", "timezone")
    ):
        result.add("current_time")
    if name in {"host_snapshot", "pressure_snapshot", "process_snapshot", "filesystem_snapshot", "service_health"} and isinstance(payload, dict):
        result.add("host_state")
    network_dict_tools = {"network_snapshot", "connection_snapshot", "local_subnets", "scan_subnet", "dns_diagnose", "network_path", "endpoint_probe", "http_probe"}
    network_list_tools = {"neighbor_snapshot", "network_reachability"}
    if (name in network_dict_tools and isinstance(payload, dict)) or (name in network_list_tools and isinstance(payload, list)):
        result.add("network_state")
    if name in {"repo_status", "repo_diff", "repo_checks", "git_status", "git_diff"} and isinstance(payload, dict):
        result.add("repository_state")
    if name == "news_search":
        payload = _json_payload(text)
        if isinstance(payload, list) and any(
            isinstance(item, dict) and str(item.get("title") or "").strip() and str(item.get("url") or "").startswith(("http://", "https://"))
            for item in payload
        ):
            result.add("news")
            result.add("web_fact")
    if name == "wiki_search":
        payload = _json_payload(text)
        if isinstance(payload, dict) and str(payload.get("summary") or "").strip():
            result.add("encyclopedic")
    if name == "market_quote":
        payload = _json_payload(text)
        quotes = payload.get("quotes") if isinstance(payload, dict) else None
        if isinstance(quotes, list) and any(isinstance(item, dict) and isinstance(item.get("price"), (int, float)) for item in quotes):
            result.add("market_price")
    if name == "browse_url" and text.strip():
        lowered = text.lower()
        browse_error = re.search(r"\b(?:404 not found|403 forbidden|access denied|page not found)\b", lowered)
        if "page returned no readable text content" not in lowered and not browse_error:
            result.add("web_fact")
    if is_weather_bearing(text) and (
        name in {"web_search", "browse_url", "run_recipe", "run_pipeline"}
        or name.startswith("recipe:")
        or "weather" in name
        or {"web_search", "browse_url"}.issubset(_recipe_stages(text))
    ):
        result.add("weather")
    return result


def make_observation(
    tool_name: str,
    content: str,
    *,
    status: str = "ok",
    at: str = "",
    turn_id: int = 0,
    arguments: Any = None,
    task_frame: dict[str, Any] | None = None,
    fact_frames: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create the minimal provenance record used by the grounding validator."""
    meta = grounding_metadata(tool_name, content, arguments=arguments, task_frame=task_frame, fact_frames=fact_frames)
    return {
        "tool": str(tool_name or ""),
        "status": str(status or ""),
        "evidence_preview": str(content or ""),
        **meta,
        "at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turn_id": max(0, int(turn_id or 0)),
    }


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fresh(item: dict[str, Any], *, now: datetime, max_age_seconds: int) -> bool:
    stamp = _parse_time(item.get("at"))
    if stamp is None:
        return False
    age = (now - stamp).total_seconds()
    return -60 <= age <= max(60, int(max_age_seconds))


def _fact_types(item: dict[str, Any]) -> set[str]:
    stored = item.get("fact_types")
    if isinstance(stored, list):
        result = {str(x) for x in stored if x}
    else:
        result = set()
    if not result:
        result = classify_fact_types(str(item.get("tool") or ""), str(item.get("evidence_preview") or item.get("content") or ""))
    return result


def _recipe_verified_payload(content: str) -> str:
    payload = _json_payload(content)
    if not isinstance(payload, dict):
        return ""
    result = payload.get("result")
    if isinstance(result, dict):
        verification = result.get("verification")
        if verification is not None:
            return json.dumps(verification, ensure_ascii=False, default=str) if not isinstance(verification, str) else verification
    if result is not None:
        return json.dumps(result, ensure_ascii=False, default=str) if not isinstance(result, str) else result
    return ""


def _weather_recipe_observation(item: dict[str, Any]) -> bool:
    tool = str(item.get("tool") or "").lower()
    stored_tools = {str(x) for x in (item.get("source_tools") or []) if x}
    recipe_like = tool.startswith("recipe:") or tool in {"run_recipe", "run_pipeline"} or "weather" in tool
    if bool(item.get("weather_verified")) and recipe_like:
        if "weather_forecast" in stored_tools or {"web_search", "browse_url"}.issubset(stored_tools):
            return True
    content = str(item.get("evidence_preview") or item.get("content") or "")
    stages = _recipe_stages(content)
    verified = _recipe_verified_payload(content)
    return bool(
        recipe_like
        and is_weather_bearing(verified)
        and ("weather_forecast" in stages or {"web_search", "browse_url"}.issubset(stages))
    )


def _weather_api_observation(item: dict[str, Any]) -> bool:
    tool = str(item.get("tool") or "").lower()
    if tool in {"web_search", "browse_url", "run_recipe", "run_pipeline"} or tool.startswith("recipe:"):
        return False
    return "weather" in tool and "weather" in _fact_types(item)


def _compact_arguments(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in list(arguments.items())[:12]:
        if isinstance(value, (str, int, float, bool)) or value is None:
            text = value if not isinstance(value, str) else value[:500]
            result[str(key)[:80]] = text
        elif isinstance(value, list):
            result[str(key)[:80]] = [str(item)[:160] for item in value[:8]]
    return result


def _canonical_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return raw.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, ""))


def _urls_from_payload(content: str) -> list[str]:
    payload = _json_payload(content)
    urls: list[str] = []
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in {"url", "href", "link"} and isinstance(child, str):
                    normalized = _canonical_url(child)
                    if normalized and normalized not in urls:
                        urls.append(normalized)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(payload)
    return urls[:20]


def _browse_source_url(content: str, arguments: dict[str, Any]) -> str:
    match = re.search(r"^URL:\s*(https?://\S+)", str(content or ""), flags=re.I | re.M)
    return _canonical_url(match.group(1) if match else str(arguments.get("url") or ""))


def _weather_query_from_recipe(content: str) -> str:
    payload = _json_payload(content)
    if not isinstance(payload, dict):
        return ""
    recovery = payload.get("grounding_recovery")
    if isinstance(recovery, dict) and recovery.get("query"):
        return str(recovery.get("query"))
    result = payload.get("result")
    if isinstance(result, dict) and result.get("query"):
        return str(result.get("query"))
    return ""


def _recipe_weather_verified(content: str) -> bool:
    payload = _json_payload(content)
    if not isinstance(payload, dict):
        return False
    result = payload.get("result")
    if not isinstance(result, dict):
        return False
    discovery = result.get("discovery")
    verification = result.get("verification")
    discovery_text = json.dumps(discovery, ensure_ascii=False, default=str)
    verification_text = verification if isinstance(verification, str) else json.dumps(verification, ensure_ascii=False, default=str)
    if not is_weather_bearing(verification_text):
        return False
    discovered = set(_urls_from_payload(discovery_text))
    verified_url = _browse_source_url(verification_text, {})
    # If the recipe returns a resolvable page URL, it must originate in discovery.
    return bool(discovered and verified_url and verified_url in discovered)


def _frame_tokens(value: Any) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 2 and token not in {"the", "city", "region", "province", "state"}
    ]


def _observation_matches_frame(item: dict[str, Any], frame: dict[str, Any], *, linked_search: bool = False) -> bool:
    if not frame or frame.get("intent") != "weather":
        return True
    entity = str(frame.get("entity") or "").strip()
    time_scope = str(frame.get("time_scope") or "").strip().lower()
    haystack = " ".join([
        str(item.get("target") or ""),
        json.dumps(item.get("arguments") or {}, ensure_ascii=False, default=str),
        str(item.get("source_url") or ""),
        # Legacy migration fallback. Modern observations also persist exact
        # lexical proof from the full untruncated result below.
        str(item.get("evidence_preview") or item.get("content") or ""),
    ]).lower()
    proof = _proof(item)
    proof_terms = {str(token).lower() for token in (proof.get("content_terms") or proof.get("scope_terms") or []) if token}
    # Exact frame-bound scope values are captured from the full result at
    # ingestion time and survive any evidence-preview clipping.
    proof_terms.update(_frame_tokens(proof.get("scope_entity") or ""))
    proof_terms.update(_frame_tokens(proof.get("scope_time") or ""))
    observed_terms = set(_frame_tokens(haystack)) | proof_terms
    entity_tokens = _frame_tokens(entity)
    tool_name = str(item.get("tool") or "").lower()
    # For browse_url, ``target`` is the source URL and is not semantic location
    # metadata. New observations carry arguments/proof; old persisted browse rows
    # often have only a source URL and remain migration-compatible.
    has_scope_metadata = (
        bool(item.get("arguments"))
        or bool(proof.get("scope_entity"))
        or bool(proof.get("scope_time"))
        or bool(proof.get("scope_terms"))
        or bool(item.get("target") and tool_name != "browse_url")
    )
    if entity_tokens and not all(token in observed_terms for token in entity_tokens):
        # Search->browse linkage proves provenance, not identity. A modern browse
        # observation must still carry the requested location/entity; otherwise a
        # result for another city in the same region could satisfy the gate.
        # Legacy observations without scope metadata remain accepted for migration.
        if has_scope_metadata:
            return False
    if time_scope and time_scope not in {"current", "now"}:
        temporal = _frame_tokens(time_scope)
        if temporal and not all(token in observed_terms for token in temporal):
            if not linked_search and has_scope_metadata:
                return False
    return True


def _news_observation_matches_frame(item: dict[str, Any], frame: dict[str, Any]) -> bool:
    """Require a scoped news query when the task names a location.

    A non-empty ``news_search`` result is not sufficient by itself: without this
    check a search for London, England or even global headlines can satisfy a
    London, Ontario request.  Modern calls carry both ``query`` and ``location``;
    older calls remain acceptable only when the query itself contains the full
    requested scope.
    """
    if not frame or frame.get("intent") != "news":
        return True
    entity = str(frame.get("entity") or "").strip()
    if not entity:
        return True
    arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
    query = str(arguments.get("query") or item.get("target") or "")
    location = str(arguments.get("location") or "")
    scope_tokens = set(_frame_tokens(" ".join((query, location))))
    segments = [segment.strip() for segment in entity.split(",") if segment.strip()]
    if not segments:
        segments = [entity]
    city_tokens = set(_frame_tokens(segments[0]))
    qualifier_tokens = set(_frame_tokens(" ".join(segments[1:])))
    if city_tokens and not city_tokens.issubset(scope_tokens):
        return False
    if qualifier_tokens and not qualifier_tokens.issubset(scope_tokens):
        return False
    return bool(scope_tokens)


def grounding_metadata(
    tool_name: str,
    content: str,
    *,
    arguments: Any = None,
    task_frame: dict[str, Any] | None = None,
    fact_frames: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return compact scope/provenance metadata safe to persist with evidence.

    ``task_frame`` is optional because tests/legacy callers may construct
    observations out of band. Runtime working-state ingestion supplies it so
    scope proof can be derived from the *full* result before preview clipping.
    """
    name = str(tool_name or "").strip().lower()
    frames = {str(k): dict(v or {}) for k, v in dict(fact_frames or {}).items() if isinstance(v, dict)}
    legacy_frame = dict(task_frame or {})

    def scoped_frame(fact_type: str) -> dict[str, Any]:
        frame = dict(frames.get(str(fact_type or "")) or {})
        if frame:
            return frame
        if str(legacy_frame.get("intent") or "") == str(fact_type or ""):
            return legacy_frame
        return legacy_frame if not frames else {}

    text = str(content or "")
    args = _compact_arguments(arguments)
    stages = sorted(_recipe_stages(text))
    facts = sorted(classify_fact_types(name, text))
    recipe_like = name.startswith("recipe:") or name in {"run_recipe", "run_pipeline"} or "weather" in name
    stage_set = set(stages)
    verified_payload = _recipe_verified_payload(text)
    weather_verified = bool(
        recipe_like
        and (
            ("weather_forecast" in stage_set and is_weather_bearing(verified_payload))
            or ({"web_search", "browse_url"}.issubset(stage_set) and _recipe_weather_verified(text))
        )
    )
    target = ""
    time_scope = ""
    source_url = ""
    discovered_urls: list[str] = []
    market_instruments: list[str] = []
    proof: dict[str, Any] = {}
    if name in {"web_search", "news_search", "wiki_search"}:
        target = str(args.get("query") or "")[:500]
        if name != "wiki_search":
            discovered_urls = _urls_from_payload(text)
            frame = scoped_frame("news" if name == "news_search" else "web_fact") or derive_task_frame(target)
            time_scope = str(frame.get("time_scope") or "")
        else:
            payload = _json_payload(text)
            if isinstance(payload, dict):
                proof["title"] = str(payload.get("title") or "")[:300]
    elif name == "browse_url":
        source_url = _browse_source_url(text, args)
        target = source_url
        # Runtime ingestion knows the current task frame. Persist exact matched
        # scope values from the full body so a requested city/time appearing late
        # in a long page cannot be lost when the human-readable preview is clipped.
        frame = scoped_frame("weather") if "weather" in facts else legacy_frame
        entity = str(frame.get("entity") or "").strip()
        entity_tokens = set(_frame_tokens(entity))
        full_terms = set(_frame_tokens(text))
        if entity_tokens and entity_tokens.issubset(full_terms):
            proof["scope_entity"] = entity[:240]
        time_value = str(frame.get("time_scope") or "").strip().lower()
        temporal = set(_frame_tokens(time_value))
        if temporal and temporal.issubset(full_terms):
            proof["scope_time"] = time_value[:120]
        # Keep a small lexical fallback for legacy/out-of-band observations that
        # do not have a task frame. It is validator-only and never prompt-rendered.
        proof["content_terms"] = _content_terms(text)
    elif name == "geocode_location":
        target = str(args.get("query") or "")[:500]
        proof["geocode_candidates"] = _geocode_candidates(text)
    elif name == "current_time":
        proof["time"] = _time_proof(text)
    elif name == "market_quote":
        market_instruments = _market_quote_instruments(text)
        proof["market_instruments"] = list(market_instruments)
    elif name == "weather_forecast":
        proof["coordinates"] = _coordinates_from_weather(text, args)
        payload = _json_payload(text)
        if isinstance(payload, dict) and payload.get("timezone"):
            proof["timezone"] = str(payload.get("timezone") or "")[:80]
    elif recipe_like:
        target = _weather_query_from_recipe(text)[:500]
        frame = scoped_frame("weather") or derive_task_frame(target)
        time_scope = str(frame.get("time_scope") or "")
        payload = _json_payload(text)
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            discovered_urls = _urls_from_payload(json.dumps(payload["result"].get("discovery"), ensure_ascii=False, default=str))
            verification = payload["result"].get("verification")
            verification_text = verification if isinstance(verification, str) else json.dumps(verification, ensure_ascii=False, default=str)
            source_url = _browse_source_url(verification_text, {})
        if target:
            proof["scope_terms"] = _content_terms(target, limit=256)
    else:
        for key in ("url", "target", "name", "host", "query"):
            if args.get(key):
                target = str(args[key])[:500]
                break
    return {
        "fact_types": facts,
        "source_tools": stages,
        "weather_verified": weather_verified,
        "arguments": args,
        "target": target,
        "time_scope": time_scope,
        "source_url": source_url,
        "discovered_urls": discovered_urls,
        "market_instruments": market_instruments,
        "grounding_proof": proof,
    }


def _explicit_evidence_reuse_request(user_request: str) -> bool:
    text = " ".join(str(user_request or "").lower().split())
    return bool(re.search(
        r"\b(?:without re-?running|without running again|previous (?:result|results|output|data|check)|"
        r"earlier (?:result|results|output|data|check)|based on (?:that|those|the previous)|"
        r"same (?:result|results|data|check)|reuse (?:that|those|the previous)|from before)\b",
        text,
    ))


def _turn_scoped_items(
    items: list[dict[str, Any]], current_turn_id: int, *, allow_carried: bool = False
) -> list[dict[str, Any]]:
    if not current_turn_id or allow_carried:
        return list(items)
    return [item for item in items if int(item.get("turn_id") or 0) == int(current_turn_id)]


def _encyclopedic_observation_matches_subject(item: dict[str, Any], subject: str) -> bool:
    subject_tokens = set(_frame_tokens(subject))
    if not subject_tokens:
        return True
    args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
    proof = _proof(item)
    explicit_scope = " ".join((
        str(args.get("query") or item.get("target") or ""),
        str(proof.get("title") or ""),
    )).strip()
    if explicit_scope:
        # Modern observations must match structured request/result scope. Do not
        # let an incidental mention in a long summary validate a wrong lookup.
        return subject_tokens.issubset(set(_frame_tokens(explicit_scope)))
    # Migration fallback for observations persisted before query/title metadata.
    legacy = str(item.get("evidence_preview") or item.get("content") or "")
    return subject_tokens.issubset(set(_frame_tokens(legacy)))


def _browse_contains_subject(item: dict[str, Any], subject: str) -> bool:
    tokens = set(_frame_tokens(subject))
    if not tokens:
        return True
    proof = _proof(item)
    observed = {str(x).lower() for x in (proof.get("content_terms") or []) if x}
    observed.update(_frame_tokens(proof.get("scope_entity") or ""))
    if tokens.issubset(observed):
        return True
    # Legacy migration fallback.
    return tokens.issubset(set(_frame_tokens(str(item.get("evidence_preview") or item.get("content") or ""))))


def _geocode_matches_entity(item: dict[str, Any], entity: str) -> bool:
    entity_tokens = set(_frame_tokens(entity))
    if not entity_tokens:
        return True
    args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
    target_tokens = set(_frame_tokens(str(args.get("query") or item.get("target") or "")))
    return entity_tokens.issubset(target_tokens)


def _linked_geocode_candidate(
    geocodes: list[dict[str, Any]], frame: dict[str, Any], *, coordinates: dict[str, Any] | None = None, timezone_name: str = ""
) -> bool:
    entity = str(frame.get("entity") or "").strip()
    if not entity:
        return True
    for item in geocodes:
        if not _geocode_matches_entity(item, entity):
            continue
        candidates = _proof(item).get("geocode_candidates") or []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if coordinates and _coordinates_close(candidate, coordinates):
                return True
            if timezone_name and str(candidate.get("timezone") or "").lower() == str(timezone_name).lower():
                return True
    return False


def _current_time_matches_frame(
    item: dict[str, Any], frame: dict[str, Any], geocodes: list[dict[str, Any]]
) -> bool:
    entity = str(frame.get("entity") or "").strip()
    if not entity:
        return True
    time_meta = _proof(item).get("time") or {}
    timezone_name = str(time_meta.get("timezone") or "")
    direct_terms = set(_frame_tokens(timezone_name))
    entity_terms = set(_frame_tokens(entity))
    if entity_terms and entity_terms.issubset(direct_terms):
        return True
    return _linked_geocode_candidate(geocodes, frame, timezone_name=timezone_name)


def _weather_api_matches_frame(
    item: dict[str, Any], frame: dict[str, Any], geocodes: list[dict[str, Any]]
) -> bool:
    if _observation_matches_frame(item, frame):
        return True
    coords = _proof(item).get("coordinates") or {}
    return bool(coords and _linked_geocode_candidate(geocodes, frame, coordinates=coords))


def validate_fact_grounding(
    user_request: str,
    observations: list[dict[str, Any]],
    *,
    current_turn_id: int = 0,
    weather_max_age_seconds: int = 10800,
    now: datetime | None = None,
    task_frame: dict[str, Any] | None = None,
    fact_frames: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Hard-check requested fact types against observation provenance.

    Weather is intentionally stricter than ordinary successful tool execution:
    the current turn needs a weather-bearing web_search + browse_url pair, a
    weather-bearing recipe/API observation, or a fresh carried weather
    observation. A current_time observation never satisfies weather.
    """
    frame = dict(task_frame or derive_task_frame(user_request))
    frames = {str(k): dict(v or {}) for k, v in dict(fact_frames or {}).items() if isinstance(v, dict)}
    required = requested_fact_types(user_request, frame, frames)
    if not frames and required:
        frames = derive_fact_frames(user_request, required_fact_types=required)

    def frame_for(fact_type: str) -> dict[str, Any]:
        scoped = dict(frames.get(str(fact_type or "")) or {})
        if scoped:
            return scoped
        if str(frame.get("intent") or "") == str(fact_type or ""):
            return frame
        return {}

    if not required:
        return {"status": "not_required", "grounded": True, "required_fact_types": [], "missing_fact_types": [], "fact_requirements": []}

    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    usable = [
        item for item in observations
        if isinstance(item, dict) and str(item.get("status") or "ok") in {"ok", "partial"}
    ]
    observed = sorted({fact for item in usable for fact in _fact_types(item)})
    missing: list[str] = []
    evidence: dict[str, list[str]] = {}
    allow_carried = _explicit_evidence_reuse_request(user_request)
    turn_usable = _turn_scoped_items(usable, current_turn_id, allow_carried=allow_carried)
    current_geocodes = [
        item for item in turn_usable
        if str(item.get("tool") or "").lower() == "geocode_location"
    ]

    if "current_time" in required:
        time_items = [
            item for item in turn_usable
            if "current_time" in _fact_types(item)
            and _current_time_matches_frame(item, frame_for("current_time"), current_geocodes)
        ]
        if time_items:
            evidence["current_time"] = [str(item.get("tool") or "") for item in time_items[-2:]]
        else:
            missing.append("current_time")

    if "encyclopedic" in required:
        encyclopedia_items = [
            item for item in usable
            if str(item.get("tool") or "").lower() == "wiki_search"
            and "encyclopedic" in _fact_types(item)
            and item in turn_usable
            and _encyclopedic_observation_matches_subject(item, encyclopedic_lookup_query(user_request))
        ]
        if encyclopedia_items:
            evidence["encyclopedic"] = ["wiki_search"]
        else:
            # Wikipedia is the preferred fast path, but a linked web discovery
            # and verified page from the current turn is an acceptable fallback
            # when the encyclopedia endpoint/package is unavailable.
            turn_items = list(turn_usable)
            subject = encyclopedic_lookup_query(user_request)
            subject_terms = set(_frame_tokens(subject))
            searches = []
            for item in turn_items:
                if str(item.get("tool") or "").lower() != "web_search":
                    continue
                query = str((item.get("arguments") or {}).get("query") or item.get("target") or "")
                query_terms = set(_frame_tokens(query))
                if not subject_terms or subject_terms.issubset(query_terms):
                    searches.append(item)
            browses = [
                item for item in turn_items
                if str(item.get("tool") or "").lower() == "browse_url" and "web_fact" in _fact_types(item)
            ]
            linked = False
            for search in searches:
                discovered = {_canonical_url(url) for url in (search.get("discovered_urls") or []) if url}
                if not discovered:
                    discovered = set(_urls_from_payload(str(search.get("evidence_preview") or search.get("content") or "")))
                for browse in browses:
                    source = _canonical_url(str(browse.get("source_url") or "")) or _browse_source_url(
                        str(browse.get("evidence_preview") or browse.get("content") or ""),
                        dict(browse.get("arguments") or {}),
                    )
                    if source and source in discovered and _browse_contains_subject(browse, subject):
                        linked = True
                        break
                if linked:
                    break
            if linked:
                evidence["encyclopedic"] = ["web_search", "browse_url"]
            else:
                missing.append("encyclopedic")

    generic_sources = {
        "host_state": {"host_snapshot", "pressure_snapshot", "process_snapshot", "filesystem_snapshot", "service_health"},
        "network_state": {"network_snapshot", "neighbor_snapshot", "connection_snapshot", "local_subnets", "scan_subnet", "network_reachability", "dns_diagnose", "network_path", "endpoint_probe", "http_probe"},
        "repository_state": {"repo_status", "repo_diff", "repo_checks", "git_status", "git_diff"},
    }
    for fact_type, allowed_tools in generic_sources.items():
        if fact_type not in required:
            continue
        matches = [
            item for item in turn_usable
            if fact_type in _fact_types(item)
            and str(item.get("tool") or "").lower() in allowed_tools
        ]
        if matches:
            evidence[fact_type] = [str(matches[-1].get("tool") or "")]
        else:
            missing.append(fact_type)

    if "news" in required:
        turn_items = list(turn_usable)
        matches = [
            item for item in turn_items
            if str(item.get("tool") or "").lower() == "news_search" and "news" in _fact_types(item)
            and _news_observation_matches_frame(item, frame_for("news"))
        ]
        if matches:
            evidence["news"] = ["news_search"]
        else:
            missing.append("news")

    if "market_price" in required:
        turn_items = list(turn_usable)
        market_frame = frame_for("market_price")
        expected = _canonical_market_instruments(market_frame.get("instruments") or extract_market_instruments(user_request))
        matches = []
        for item in turn_items:
            if str(item.get("tool") or "").lower() != "market_quote" or "market_price" not in _fact_types(item):
                continue
            # Prefer compact structured scope captured from the *full* provider
            # payload before the evidence preview is bounded. This survives
            # working-state clipping and proves returned rows, unlike call args.
            actual = _canonical_market_instruments(item.get("market_instruments") or [])
            if actual:
                if expected and not expected.issubset(actual):
                    continue
                matches.append(item)
                continue
            payload = _json_payload(str(item.get("content") or item.get("evidence_preview") or ""))
            if not isinstance(payload, dict):
                # Legacy observations did not persist returned instrument scope.
                # Arguments are acceptable only for a single requested asset:
                # with multiple assets they cannot prove a partial provider
                # response actually contained every requested quote.
                args = dict(item.get("arguments") or {})
                actual_args = _canonical_market_instruments(args.get("instruments") or [])
                if len(expected) != 1 or not actual_args or not expected.issubset(actual_args):
                    continue
                matches.append(item)
                continue
            rows = [row for row in (payload.get("quotes") or []) if isinstance(row, dict) and isinstance(row.get("price"), (int, float))]
            actual = _canonical_market_instruments(
                [value for row in rows for value in (row.get("instrument"), row.get("symbol"))]
            )
            if expected and not expected.issubset(actual):
                continue
            matches.append(item)
        if matches:
            evidence["market_price"] = ["market_quote"]
        else:
            missing.append("market_price")

    if "web_fact" in required:
        turn_items = list(turn_usable)
        explicit_urls = {
            _canonical_url(url) for url in re.findall(r"https?://[^\s<>\"']+", str(user_request or ""), flags=re.I)
        }
        explicit_urls.discard("")
        stop = {
            "search", "web", "look", "current", "verify", "source", "sources", "official",
            "documentation", "docs", "find", "about", "please", "tell", "show", "with", "from",
            "that", "this", "what", "who", "where", "when", "latest", "the", "and", "for",
            "into", "onto", "are", "was", "were", "has", "have", "had", "can", "could",
        }
        request_terms = {
            token for token in re.findall(r"[a-z0-9]+", str(user_request or "").lower())
            if len(token) > 2 and token not in stop
        }
        searches = []
        for item in turn_items:
            if str(item.get("tool") or "").lower() != "web_search":
                continue
            query = str((item.get("arguments") or {}).get("query") or item.get("target") or "").lower()
            query_terms = set(re.findall(r"[a-z0-9]+", query))
            overlap = len(request_terms & query_terms)
            needed = min(2, len(request_terms))
            if not request_terms or overlap >= needed:
                searches.append(item)
        browses = [
            item for item in turn_items
            if str(item.get("tool") or "").lower() == "browse_url" and "web_fact" in _fact_types(item)
        ]
        qualified: list[dict[str, Any]] = []
        if explicit_urls:
            for browse in browses:
                source = _canonical_url(str(browse.get("source_url") or "")) or _browse_source_url(
                    str(browse.get("evidence_preview") or browse.get("content") or ""), dict(browse.get("arguments") or {})
                )
                if source in explicit_urls:
                    qualified.append(browse)
        else:
            for search in searches:
                discovered = {_canonical_url(url) for url in (search.get("discovered_urls") or []) if url}
                if not discovered:
                    discovered = set(_urls_from_payload(str(search.get("evidence_preview") or search.get("content") or "")))
                for browse in browses:
                    source = _canonical_url(str(browse.get("source_url") or "")) or _browse_source_url(
                        str(browse.get("evidence_preview") or browse.get("content") or ""), dict(browse.get("arguments") or {})
                    )
                    if source and source in discovered:
                        qualified.append(browse)
            # Backward compatibility for persisted observations created before
            # arguments/provenance were recorded. New runtime observations carry
            # arguments and therefore must satisfy discovery->verification linkage.
            if not qualified:
                legacy = [item for item in browses if not item.get("arguments") and not item.get("discovered_urls")]
                qualified.extend(legacy)
        if qualified:
            evidence["web_fact"] = ["web_search", "browse_url"] if searches else ["browse_url"]
        else:
            missing.append("web_fact")

    if "weather" in required:
        current = _turn_scoped_items(usable, current_turn_id, allow_carried=False)
        current_weather = [item for item in current if "weather" in _fact_types(item)]
        current_search = [
            item for item in current_weather
            if str(item.get("tool") or "").lower() == "web_search"
            and _observation_matches_frame(item, frame_for("weather"))
        ]
        current_recipe = [
            item for item in current_weather
            if _weather_recipe_observation(item) and _observation_matches_frame(item, frame_for("weather"))
        ]
        current_api = [
            item for item in current_weather
            if _weather_api_observation(item)
            and _weather_api_matches_frame(item, frame_for("weather"), [
                geo for geo in current
                if str(geo.get("tool") or "").lower() == "geocode_location"
            ])
        ]

        linked_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for search in current_search:
            discovered = {_canonical_url(url) for url in (search.get("discovered_urls") or []) if url}
            if not discovered:
                # Forward-fill metadata for observations created by older versions.
                discovered = set(_urls_from_payload(str(search.get("evidence_preview") or search.get("content") or "")))
            for browse in current_weather:
                if str(browse.get("tool") or "").lower() != "browse_url":
                    continue
                source = _canonical_url(str(browse.get("source_url") or ""))
                if not source:
                    source = _browse_source_url(str(browse.get("evidence_preview") or browse.get("content") or ""), dict(browse.get("arguments") or {}))
                if source and source in discovered and _observation_matches_frame(browse, frame_for("weather"), linked_search=True):
                    linked_pairs.append((search, browse))

        stored_weather = [
            item for item in usable
            if "weather" in _fact_types(item)
            and (not current_turn_id or int(item.get("turn_id") or 0) != int(current_turn_id))
            and str(item.get("tool") or "").lower() != "web_search"
            and _fresh(item, now=now_utc, max_age_seconds=weather_max_age_seconds)
            and _observation_matches_frame(item, frame_for("weather"))
            and (_weather_recipe_observation(item) or _weather_api_observation(item) or str(item.get("tool") or "").lower() == "browse_url")
        ]

        if current_api:
            evidence["weather"] = [str(current_api[-1].get("tool") or "weather API")]
        elif current_recipe:
            evidence["weather"] = [str(current_recipe[-1].get("tool") or "weather recipe")]
        elif linked_pairs:
            evidence["weather"] = ["web_search", "browse_url"]
        elif stored_weather:
            evidence["weather"] = [f"stored:{str(stored_weather[-1].get('tool') or 'weather')}" ]
        else:
            missing.append("weather")

    fact_requirements = [
        {
            "fact_type": fact_type,
            "status": "satisfied" if fact_type in evidence else "pending",
            "satisfied": fact_type in evidence,
            "evidence": list(evidence.get(fact_type) or []),
            "last_error": "" if fact_type in evidence else "insufficient_evidence",
        }
        for fact_type in sorted(required)
    ]

    if missing:
        return {
            "status": "missing_evidence",
            "grounded": False,
            "required_fact_types": sorted(required),
            "missing_fact_types": sorted(set(missing)),
            "observed_fact_types": observed,
            "evidence": evidence,
            "diagnosis": "insufficient_evidence",
            "reason": "requested fact type is not present in qualifying observations",
            "fact_requirements": fact_requirements,
        }
    return {
        "status": "grounded",
        "grounded": True,
        "required_fact_types": sorted(required),
        "missing_fact_types": [],
        "observed_fact_types": observed,
        "evidence": evidence,
        "diagnosis": "task_complete",
        "fact_requirements": fact_requirements,
    }


def _location_hint(memory_context: str) -> str:
    """Extract an explicitly stored location from mixed profile/memory context."""
    raw = str(memory_context or "")
    if not raw.strip():
        return ""

    # Fast path for a pure JSON memory payload.
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        payload = None
    rows = payload if isinstance(payload, list) else ([payload] if isinstance(payload, dict) else [])
    for item in rows:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or item.get("key") or "").lower()
        if not any(token in topic for token in ("location", "city", "home")):
            continue
        fact = str(item.get("fact") or item.get("value") or item.get("content") or "").strip()
        if fact:
            return re.sub(r"\s+", " ", fact)[:180]

    # Normal turn context combines a rendered profile block and JSON memories,
    # so json.loads() on the whole string is intentionally not required.
    match = re.search(r"\*\*Location\*\*:\s*([^\n]+)", raw, flags=re.I)
    if match:
        return re.sub(r"\s+", " ", match.group(1).strip())[:180]
    match = re.search(
        r'"topic"\s*:\s*"(?:user_)?(?:location|city|home)[^"]*"[^{}]{0,500}?"fact"\s*:\s*"([^"]+)"',
        raw, flags=re.I | re.S,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1).strip())[:180]
    return ""


def _weather_location(user_request: str, memory_context: str = "") -> str:
    """Resolve only explicit/requested or explicitly stored weather location."""
    frame = derive_task_frame(user_request)
    entity = str(frame.get("entity") or "").strip()
    if entity:
        return entity[:180]
    hint = _location_hint(memory_context)
    if hint:
        return hint
    # The profile owns the canonical declared location; use it directly rather
    # than depending on semantic-memory retrieval to happen to surface it.
    try:
        from .user_profile import get_user_location
        return str(get_user_location() or "").strip()[:180]
    except Exception:
        return ""


def _forecast_days_for_request(user_request: str) -> int:
    """Return a bounded provider horizon that fully covers the requested period."""
    text = str(user_request or "").lower()
    if re.search(r"\b(?:today|tonight|now|currently|right now|this (?:morning|afternoon|evening))\b|\bat the moment\b", text):
        return 1
    if re.search(r"\btomorrow\b", text):
        return 2  # provider includes today
    match = re.search(r"\bnext\s+(\d{1,2})\s+days?\b", text)
    if match:
        return max(2, min(int(match.group(1)) + 1, 16))
    if re.search(r"\bnext week\b|\b(?:7|seven)[ -]?day\b|\bweek(?:ly)? forecast\b", text):
        return 8  # today + seven future days
    if re.search(r"\bweekend\b", text):
        return 8
    return 8


def build_weather_query(user_request: str, memory_context: str = "") -> str:
    request = re.sub(r"\s+", " ", str(user_request or "")).strip()
    hint = _weather_location(request, memory_context)
    if hint and hint.lower() not in request.lower():
        request = f"{request} {hint}"
    if not _WEATHER_PRIMARY_RE.search(request):
        request = f"weather forecast {request}".strip()
    return request[:1000]


def weather_structured_stages() -> list[dict[str, Any]]:
    """Deterministic keyless weather recipe: geocode place -> structured forecast."""
    return [
        {"id": "place", "tool": "geocode_location", "args": {
            "query": {"$param": "location"}, "count": 1,
        }},
        {"id": "forecast", "tool": "weather_forecast", "args": {
            "latitude": {"$ref": "place", "path": "0.latitude"},
            "longitude": {"$ref": "place", "path": "0.longitude"},
            "forecast_days": {"$param": "forecast_days", "default": 8},
            "timezone_name": "auto",
        }},
        {"id": "result", "tool": "compose_object", "args": {"data": {
            "location": {"$param": "location"},
            "place": {"$ref": "place", "path": "0"},
            "forecast": {"$ref": "forecast"},
        }}},
    ]


def weather_fallback_stages() -> list[dict[str, Any]]:
    """Independent web-search fallback used only when structured weather fails."""
    return [
        {"id": "search", "tool": "web_search", "args": {"query": {"$param": "query", "default": "current weather forecast"}}},
        {"id": "verify", "tool": "browse_url", "args": {"url": {"$ref": "search", "path": "0.url"}}},
        {"id": "result", "tool": "compose_object", "args": {"data": {
            "query": {"$param": "query", "default": "current weather forecast"},
            "discovery": {"$ref": "search"},
            "verification": {"$ref": "verify"},
        }}},
    ]


def execute_weather_grounding_recovery(
    user_request: str,
    memory_context: str = "",
    *,
    frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch structured weather first, then fall back to search + verified page."""
    from .pipeline import execute_pipeline
    from .recipe_store import get_recipe

    scoped_frame = dict(frame or {})
    scoped_request = str(scoped_frame.get("source_text") or user_request or "")
    location = str(scoped_frame.get("entity") or "").strip() or _weather_location(scoped_request, memory_context)
    time_scope = str(scoped_frame.get("time_scope") or "").strip()
    query_request = " ".join(part for part in (scoped_request, location, time_scope) if part)
    query = build_weather_query(query_request, memory_context)
    forecast_days = _forecast_days_for_request(" ".join(part for part in (scoped_request, time_scope) if part))
    structured_error = ""

    if location:
        try:
            recipe = get_recipe(WEATHER_RECIPE_NAME)
        except Exception:
            recipe = None
        stages = list(recipe.get("pipeline") or []) if recipe else weather_structured_stages()
        result = execute_pipeline(stages, {
            "location": location,
            "forecast_days": forecast_days,
            "query": query,
        })
        result["grounding_recovery"] = {
            "fact_type": "weather",
            "query": query,
            "location": location,
            "forecast_days": forecast_days,
            "source": "structured_weather_recipe" if recipe else "structured_weather_fallback",
            "recipe": WEATHER_RECIPE_NAME if recipe else "",
        }
        if bool(result.get("ok")):
            return result
        structured_error = str(result.get("error") or "structured weather recovery failed")[:1000]
    else:
        structured_error = "no explicit or stored location is available"

    fallback = execute_pipeline(weather_fallback_stages(), {"query": query})
    fallback["grounding_recovery"] = {
        "fact_type": "weather",
        "query": query,
        "location": location,
        "forecast_days": forecast_days,
        "source": "web_verification_fallback",
        "recipe": WEATHER_RECIPE_NAME,
        "structured_error": structured_error,
    }
    return fallback
