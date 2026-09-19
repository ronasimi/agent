"""Deterministic fact-type grounding gates for final-answer control.

The main/fast models may decide *how* to recover, but they cannot declare a
fact-retrieval turn grounded.  This module derives a small set of requested fact
types from the user request and checks trusted tool provenance plus bounded
content signals before finalization.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

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
    r"\b(?:network (?:interfaces|routes|health|state|connections?)|listening sockets?|active connections?|neighbor table|arp|ndp)\b",
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
_WEATHER_PRIMARY_RE = re.compile(r"\b(weather|forecast|current conditions?)\b", re.I)
_WEATHER_DETAIL_RE = re.compile(
    r"(?:\btemperature\b|\btemp\b|\bhighs?\b|\blows?\b|\bhumidity\b|\bwind(?:s|y)?\b|"
    r"\bprecipitation\b|\brain(?:fall|y)?\b|\bsnow(?:fall|y)?\b|\bfeels like\b|\bdew point\b|"
    r"\bchance of\b|\bvisibility\b|\bpressure\b|°\s*[CF]\b)",
    re.I,
)


def requested_fact_types(user_request: str) -> set[str]:
    """Return fact types with deterministic grounding policies for this request.

    The patterns intentionally describe retrieval intents, not arbitrary keyword
    mentions, so editing code that happens to contain the word ``weather`` does
    not trigger a live-weather grounding requirement.
    """
    text = " ".join(str(user_request or "").split())
    result: set[str] = set()
    if any(pattern.search(text) for pattern in _WEATHER_REQUEST_PATTERNS):
        result.add("weather")
    if _TIME_REQUEST_RE.search(text):
        result.add("current_time")
    if _HOST_REQUEST_RE.search(text):
        result.add("host_state")
    if _NETWORK_REQUEST_RE.search(text):
        result.add("network_state")
    if _REPO_REQUEST_RE.search(text):
        result.add("repository_state")
    if _WEB_FACT_REQUEST_RE.search(text) and not result:
        result.add("web_fact")
    return result


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
        if isinstance(item, dict) and item.get("ok", True)
    }


def classify_fact_types(tool_name: str, content: str) -> set[str]:
    """Classify fact types carried by one successful observation."""
    name = str(tool_name or "").strip().lower()
    text = str(content or "")
    result: set[str] = set()
    if name == "current_time" and re.search(r"\d{1,4}[-/:T ]\d{1,2}|timezone|utc|local", text, re.I):
        result.add("current_time")
    if name in {"host_snapshot", "pressure_snapshot", "process_snapshot", "filesystem_snapshot", "service_health"} and text.strip():
        result.add("host_state")
    if name in {"network_snapshot", "neighbor_snapshot", "connection_snapshot", "local_subnets", "scan_subnet", "network_reachability", "dns_diagnose", "network_path", "endpoint_probe", "http_probe"} and text.strip():
        result.add("network_state")
    if name in {"repo_status", "repo_diff", "repo_checks", "git_status", "git_diff"} and text.strip():
        result.add("repository_state")
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
) -> dict[str, Any]:
    """Create the minimal provenance record used by the grounding validator."""
    meta = grounding_metadata(tool_name, content)
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
    if bool(item.get("weather_verified")) and {"web_search", "browse_url"}.issubset(stored_tools):
        return tool.startswith("recipe:") or tool in {"run_recipe", "run_pipeline"} or "weather" in tool
    content = str(item.get("evidence_preview") or item.get("content") or "")
    stages = _recipe_stages(content)
    verified = _recipe_verified_payload(content)
    return (
        (tool.startswith("recipe:") or tool in {"run_recipe", "run_pipeline"} or "weather" in tool)
        and {"web_search", "browse_url"}.issubset(stages)
        and is_weather_bearing(verified)
    )


def _weather_api_observation(item: dict[str, Any]) -> bool:
    tool = str(item.get("tool") or "").lower()
    if tool in {"web_search", "browse_url", "run_recipe", "run_pipeline"} or tool.startswith("recipe:"):
        return False
    return "weather" in tool and "weather" in _fact_types(item)


def grounding_metadata(tool_name: str, content: str) -> dict[str, Any]:
    """Return compact provenance metadata safe to persist beside clipped evidence."""
    name = str(tool_name or "").strip().lower()
    text = str(content or "")
    stages = sorted(_recipe_stages(text))
    facts = sorted(classify_fact_types(name, text))
    recipe_like = name.startswith("recipe:") or name in {"run_recipe", "run_pipeline"} or "weather" in name
    weather_verified = bool(
        recipe_like
        and {"web_search", "browse_url"}.issubset(set(stages))
        and is_weather_bearing(_recipe_verified_payload(text))
    )
    return {
        "fact_types": facts,
        "source_tools": stages,
        "weather_verified": weather_verified,
    }


def validate_fact_grounding(
    user_request: str,
    observations: list[dict[str, Any]],
    *,
    current_turn_id: int = 0,
    weather_max_age_seconds: int = 10800,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Hard-check requested fact types against observation provenance.

    Weather is intentionally stricter than ordinary successful tool execution:
    the current turn needs a weather-bearing web_search + browse_url pair, a
    weather-bearing recipe/API observation, or a fresh carried weather
    observation. A current_time observation never satisfies weather.
    """
    required = requested_fact_types(user_request)
    if not required:
        return {"status": "not_required", "grounded": True, "required_fact_types": [], "missing_fact_types": []}

    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    usable = [
        item for item in observations
        if isinstance(item, dict) and str(item.get("status") or "ok") in {"ok", "partial"}
    ]
    observed = sorted({fact for item in usable for fact in _fact_types(item)})
    missing: list[str] = []
    evidence: dict[str, list[str]] = {}

    if "current_time" in required:
        time_items = [item for item in usable if "current_time" in _fact_types(item)]
        if time_items:
            evidence["current_time"] = [str(item.get("tool") or "") for item in time_items[-2:]]
        else:
            missing.append("current_time")

    generic_sources = {
        "host_state": {"host_snapshot", "pressure_snapshot", "process_snapshot", "filesystem_snapshot", "service_health"},
        "network_state": {"network_snapshot", "neighbor_snapshot", "connection_snapshot", "local_subnets", "scan_subnet", "network_reachability", "dns_diagnose", "network_path", "endpoint_probe", "http_probe"},
        "repository_state": {"repo_status", "repo_diff", "repo_checks", "git_status", "git_diff"},
        "web_fact": {"browse_url"},
    }
    for fact_type, allowed_tools in generic_sources.items():
        if fact_type not in required:
            continue
        matches = [
            item for item in usable
            if fact_type in _fact_types(item)
            and str(item.get("tool") or "").lower() in allowed_tools
        ]
        if matches:
            evidence[fact_type] = [str(matches[-1].get("tool") or "")]
        else:
            missing.append(fact_type)

    if "weather" in required:
        current = [
            item for item in usable
            if current_turn_id and int(item.get("turn_id") or 0) == int(current_turn_id)
        ]
        if not current_turn_id:
            current = usable
        current_weather = [item for item in current if "weather" in _fact_types(item)]
        current_search = [item for item in current_weather if str(item.get("tool") or "").lower() == "web_search"]
        current_browse = [item for item in current_weather if str(item.get("tool") or "").lower() == "browse_url"]
        current_recipe = [item for item in current_weather if _weather_recipe_observation(item)]
        current_api = [item for item in current_weather if _weather_api_observation(item)]

        # Carried observations are acceptable only while fresh. Prefer verified
        # page/API/recipe evidence; search snippets alone are not stored-weather
        # verification.
        stored_weather = [
            item for item in usable
            if "weather" in _fact_types(item)
            and (not current_turn_id or int(item.get("turn_id") or 0) != int(current_turn_id))
            and str(item.get("tool") or "").lower() != "web_search"
            and _fresh(item, now=now_utc, max_age_seconds=weather_max_age_seconds)
        ]

        if current_api:
            evidence["weather"] = [str(current_api[-1].get("tool") or "weather API")]
        elif current_recipe:
            evidence["weather"] = [str(current_recipe[-1].get("tool") or "weather recipe")]
        elif current_search and current_browse:
            evidence["weather"] = ["web_search", "browse_url"]
        elif stored_weather:
            evidence["weather"] = [f"stored:{str(stored_weather[-1].get('tool') or 'weather')}" ]
        else:
            missing.append("weather")

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
        }
    return {
        "status": "grounded",
        "grounded": True,
        "required_fact_types": sorted(required),
        "missing_fact_types": [],
        "observed_fact_types": observed,
        "evidence": evidence,
        "diagnosis": "task_complete",
    }


def _location_hint(memory_context: str) -> str:
    """Extract only an explicitly stored location-like memory for weather search."""
    try:
        payload = json.loads(str(memory_context or ""))
    except (TypeError, json.JSONDecodeError):
        return ""
    rows = payload if isinstance(payload, list) else [payload]
    for item in rows:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or item.get("key") or "").lower()
        if not any(token in topic for token in ("location", "city", "home")):
            continue
        fact = str(item.get("fact") or item.get("value") or item.get("content") or "").strip()
        if fact:
            return re.sub(r"\s+", " ", fact)[:180]
    return ""


def build_weather_query(user_request: str, memory_context: str = "") -> str:
    request = re.sub(r"\s+", " ", str(user_request or "")).strip()
    hint = _location_hint(memory_context)
    if hint and hint.lower() not in request.lower():
        request = f"{request} {hint}"
    if not _WEATHER_PRIMARY_RE.search(request):
        request = f"weather forecast {request}".strip()
    return request[:1000]


def weather_fallback_stages() -> list[dict[str, Any]]:
    """Return the deterministic primitive fallback used if the saved recipe is unavailable."""
    return [
        {"id": "search", "tool": "web_search", "args": {"query": {"$param": "query", "default": "current weather forecast"}}},
        {"id": "verify", "tool": "browse_url", "args": {"url": {"$ref": "search", "path": "0.url"}}},
        {"id": "result", "tool": "compose_object", "args": {"data": {
            "query": {"$param": "query", "default": "current weather forecast"},
            "discovery": {"$ref": "search"},
            "verification": {"$ref": "verify"},
        }}},
    ]


def execute_weather_grounding_recovery(user_request: str, memory_context: str = "") -> dict[str, Any]:
    """Execute the builtin weather recipe, falling back to its primitive chain."""
    from .pipeline import execute_pipeline
    from .recipe_store import get_recipe

    query = build_weather_query(user_request, memory_context)
    try:
        recipe = get_recipe(WEATHER_RECIPE_NAME)
    except Exception:
        recipe = None
    stages = list(recipe.get("pipeline") or []) if recipe else weather_fallback_stages()
    result = execute_pipeline(stages, {"query": query})
    result["grounding_recovery"] = {
        "fact_type": "weather",
        "query": query,
        "source": "builtin_recipe" if recipe else "primitive_fallback",
        "recipe": WEATHER_RECIPE_NAME if recipe else "",
    }
    return result
