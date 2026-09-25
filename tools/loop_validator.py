"""Compatibility parsers for legacy tool outcomes; no model calls or routing."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# Only deterministic, leading error markers are treated as failures.  Avoid
# scanning arbitrary tool text for words such as "error" because fetched pages,
# logs, and source files routinely contain those words as data.
_ERROR_PREFIXES = (
    "error:",
    "error reading ",
    "error writing ",
    "error browsing ",
    "tool execution error:",
    "execution error:",
    "python execution error:",
    "web search error:",
    "wikipedia search error:",
    "package search error:",
    "package installation error:",
    "package installation failed",
    "failed to ",
    "failed:",
    "validation error:",
    "mdns scan encountered an error:",
    "embedding generation failed:",
    "ollama health check failed:",
    "notification failed",
    "notification error:",
    "search execution failed:",
)

_SOFT_FAILURE_PREFIXES = (
    "no search results found",
    "no official packages found",
    "no active hosts discovered",
    "no mdns services discovered",
    "no related memories found",
    "no logs found",
)

_PARTIAL_PREFIXES = (
    "partial:",
)

# These tools intentionally encode a negative diagnostic outcome as structured
# data (for example connection refused / DNS failure). The tool invocation still
# succeeded and the negative state is often the fact the user requested.
_STRUCTURED_NEGATIVE_DIAGNOSTIC_TOOLS = {
    "endpoint_probe", "http_probe", "tcp_connect", "tls_handshake",
}

# Empty/negative text from these tools is a completed observation, not an
# execution failure. Search/memory misses remain no-progress because callers can
# usually recover by changing the query/source.
_VALID_EMPTY_RESULT_PREFIXES = {
    "read_host_journal": ("no logs found",),
    "scan_mdns": ("no mdns services discovered",),
    "search_packages": ("no official packages found",),
    "scan_subnet": ("error: no active hosts discovered",),
}

# Deterministic primitives whose successful contract is structured JSON.  If one
# of these returns malformed/unexpected top-level data, treating arbitrary text
# as success can satisfy a requirement while leaving the grounding layer with no
# usable evidence.  Negative diagnostic states are still valid JSON and remain
# successful observations.
_JSON_LIST_RESULT_TOOLS = {
    "web_search", "news_search", "geocode_location", "neighbor_snapshot",
    "network_reachability",
}
_JSON_DICT_RESULT_TOOLS = {
    "wiki_search", "market_quote", "current_time", "weather_forecast",
    "host_snapshot", "pressure_snapshot", "process_snapshot",
    "filesystem_snapshot", "service_health", "network_snapshot",
    "connection_snapshot", "local_subnets", "scan_subnet", "dns_diagnose",
    "network_path", "endpoint_probe", "http_probe",
    "repo_status", "repo_diff", "repo_checks", "git_status", "git_diff",
    "start_computation", "get_computation_status", "cancel_computation",
    "browser_step",
}


def tool_call_signature(call: dict[str, Any]) -> str:
    """Return a stable signature used to reject repeated recovery calls."""
    function = call.get("function", {}) if isinstance(call, dict) else {}
    name = str(function.get("name") or "")
    arguments = function.get("arguments", {})
    try:
        encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = str(arguments)
    return f"{name}:{encoded}"


def result_fingerprint(text: str) -> str:
    """Return a small stable digest for detecting identical no-progress results."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:16]


def classify_tool_outcome(
    content: str,
    *,
    tool_name: str = "",
    execution_error: bool = False,
    media_error: bool = False,
) -> dict[str, str | bool]:
    """Classify only explicit harness/tool failures; ordinary data remains successful."""
    text = str(content or "").strip()
    if execution_error:
        return {"success": False, "status": "error", "reason": "execution_error", "fingerprint": result_fingerprint(text)}
    if media_error:
        return {"success": False, "status": "error", "reason": "media_attachment_failed", "fingerprint": result_fingerprint(text)}

    lowered = text.lower().lstrip()
    name = str(tool_name or "")

    # browser_step returns structured machine-readable failures rather than an
    # English "Error:" prefix. Feed its normalized code directly into the
    # existing stall tracker/validator so retries can pivot on the actual UI
    # failure class.
    if name == "browser_step" and text.startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("ok") is False:
                error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
                code = re.sub(r"[^a-z0-9_]+", "_", str(error.get("code") or "browser_error").lower()).strip("_")
                return {"success": False, "status": "error", "reason": f"browser_{code}", "fingerprint": result_fingerprint(text)}
            if isinstance(payload, dict) and payload.get("ok") is True:
                return {"success": True, "status": "ok", "reason": "ok", "fingerprint": result_fingerprint(text)}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {"success": False, "status": "error", "reason": "malformed_structured_result", "fingerprint": result_fingerprint(text)}

    # Recognize explicit textual control/status prefixes before enforcing a
    # structured success schema. Otherwise a legitimate ``Error: ...`` emitted
    # by a JSON-returning primitive is mislabeled as ``malformed_structured_result``
    # and can send the model into argument-tweaking retries. Tool-specific valid
    # empty states take precedence over the generic ``error:`` prefix.
    for prefix in _VALID_EMPTY_RESULT_PREFIXES.get(name, ()):
        if lowered.startswith(prefix):
            return {"success": True, "status": "ok", "reason": "empty_result", "fingerprint": result_fingerprint(text)}
    if lowered.startswith("error: reminder backend unavailable:"):
        return {"success": False, "status": "error", "reason": "tool_unavailable", "fingerprint": result_fingerprint(text)}
    if lowered.startswith(_ERROR_PREFIXES):
        return {"success": False, "status": "error", "reason": "tool_reported_error", "fingerprint": result_fingerprint(text)}
    if lowered.startswith(_SOFT_FAILURE_PREFIXES):
        return {"success": False, "status": "error", "reason": "no_progress_result", "fingerprint": result_fingerprint(text)}

    # A few read tools have deterministic empty/invalid-result shapes that otherwise
    # look like successful JSON/text. Mark only retrieval failures as no-progress;
    # diagnostic tools may legitimately return negative states (for example ok=false
    # when an endpoint is unreachable) and those remain useful evidence.
    if name in {"web_search", "news_search"} and text.startswith("["):
        try:
            payload = json.loads(text)
            valid_rows = [
                item for item in payload
                if isinstance(item, dict)
                and str(item.get("title") or "").strip()
                and str(item.get("url") or "").startswith(("http://", "https://"))
            ] if isinstance(payload, list) else []
            # The news primitive already performs its one bounded broader-window
            # fallback internally. A valid [] therefore means "provider returned
            # no qualifying headlines", which is a completed observation rather
            # than a malformed/failing invocation. It still carries no news fact
            # grounding, so callers may only report the retrieval miss.
            if name == "news_search" and isinstance(payload, list) and not payload:
                return {"success": True, "status": "ok", "reason": "empty_result", "fingerprint": result_fingerprint(text)}
            if not valid_rows:
                return {"success": False, "status": "error", "reason": "no_progress_result", "fingerprint": result_fingerprint(text)}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if name == "wiki_search" and text.startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and not str(payload.get("summary") or "").strip():
                return {"success": False, "status": "error", "reason": "no_progress_result", "fingerprint": result_fingerprint(text)}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if name == "market_quote" and text.startswith("{"):
        try:
            payload = json.loads(text)
            quotes = payload.get("quotes") if isinstance(payload, dict) else None
            numeric = [
                item for item in (quotes or [])
                if isinstance(item, dict) and isinstance(item.get("price"), (int, float))
            ]
            if not numeric:
                return {"success": False, "status": "error", "reason": "no_progress_result", "fingerprint": result_fingerprint(text)}
            if isinstance(payload, dict) and payload.get("errors"):
                return {"success": True, "status": "partial", "reason": "partial_result", "fingerprint": result_fingerprint(text)}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if name == "browse_url" and "the page returned no readable text content." in lowered:
        return {"success": False, "status": "error", "reason": "no_progress_result", "fingerprint": result_fingerprint(text)}
    if name in (_JSON_LIST_RESULT_TOOLS | _JSON_DICT_RESULT_TOOLS):
        try:
            structured = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            structured = None
        expected_type = list if name in _JSON_LIST_RESULT_TOOLS else dict
        if not isinstance(structured, expected_type):
            return {"success": False, "status": "error", "reason": "malformed_structured_result", "fingerprint": result_fingerprint(text)}
        if name == "current_time" and not all(str(structured.get(key) or "").strip() for key in ("utc", "local", "timezone")):
            return {"success": False, "status": "error", "reason": "malformed_structured_result", "fingerprint": result_fingerprint(text)}
        if name == "geocode_location":
            valid_locations = [
                row for row in structured
                if isinstance(row, dict)
                and isinstance(row.get("latitude"), (int, float))
                and isinstance(row.get("longitude"), (int, float))
            ]
            if not valid_locations:
                return {"success": False, "status": "error", "reason": "no_progress_result", "fingerprint": result_fingerprint(text)}
        if name == "weather_forecast" and not (
            isinstance(structured.get("daily"), dict)
            and isinstance(structured.get("latitude"), (int, float))
            and isinstance(structured.get("longitude"), (int, float))
        ):
            return {"success": False, "status": "error", "reason": "malformed_structured_result", "fingerprint": result_fingerprint(text)}

    # Tools that return JSON can expose a top-level error without using a string
    # prefix.  Only inspect the top level so untrusted nested data is not treated
    # as harness control information. Structured connectivity probes are an
    # exception: ok=false + error describes the observed endpoint state.
    if text.startswith("{"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("error"):
                if name in _STRUCTURED_NEGATIVE_DIAGNOSTIC_TOOLS and payload.get("ok") is False:
                    return {"success": True, "status": "ok", "reason": "diagnostic_negative", "fingerprint": result_fingerprint(text)}
                return {"success": False, "status": "error", "reason": "tool_reported_error", "fingerprint": result_fingerprint(text)}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    if lowered.startswith(_PARTIAL_PREFIXES):
        # Execution tools reporting a non-zero exit code may contain useful
        # diagnostic stdout, but the requested action did not succeed. Do not
        # let that output satisfy a mutating/completion requirement.
        if name in {"execute_shell", "execute_python", "install_package"}:
            return {"success": False, "status": "error", "reason": "nonzero_exit", "fingerprint": result_fingerprint(text)}
        return {"success": True, "status": "partial", "reason": "nonzero_with_output", "fingerprint": result_fingerprint(text)}
    return {"success": True, "status": "ok", "reason": "ok", "fingerprint": result_fingerprint(text)}
