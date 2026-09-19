"""Deterministic task requirements for broad multi-tool requests.

The ledger is intentionally conservative: it only creates requirements from
explicit phrases in the current user request.  It is used for tool exposure and
a pre-final completeness gate; it does not infer conclusions from tool output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


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
        }


_RULES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
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
    ("http", "http_probe", "HTTP/HTTPS connectivity probe", (r"\bprobe (?:https?|connectivity)", r"\bhttps? connectivity\b", r"\bhttp probe\b")),
    ("path", "network_path", "network path/hop diagnosis", (r"\bnetwork path\b", r"\btraceroute\b", r"\bmtr\b", r"\broute tracing\b")),
    ("tool_health", "tool_health", "registered tool/dependency health", (r"\btool/?dependency health\b", r"\btool health\b", r"\bcurrent tool.*health\b")),
    ("dependency_audit", "dependency_audit", "runtime dependency audit", (r"\bdependency health\b", r"\bdependency audit\b", r"\btool/?dependency health\b")),
    ("news_search", "news_search", "current news headline discovery", (r"\b(?:latest|recent|current|today(?:'s)?)\b.{0,48}\b(?:news|headlines?|stories?)\b", r"\b(?:latest|top|local)\s+headlines?\b")),
    ("web_search", "web_search", "current web source discovery", (r"\bweb research\b", r"\bresearch the current\b", r"\bcurrent .*documentation\b", r"\blook up\b", r"\bsearch the web\b")),
    ("web_verify", "browse_url", "authoritative source content verification", (r"\bcurrent .*documentation\b", r"\bofficial .*documentation\b", r"\bsource urls?\b", r"\bverify .*source\b", r"\bweb research\b")),
    ("screenshot", "take_web_screenshot", "requested webpage screenshot", (r"\btake (?:a )?screenshot\b", r"\bscreenshot of\b", r"\bcapture .*page\b")),
    ("repo_status", "repo_status", "repository status", (r"\brepository status\b", r"\brepo status\b")),
    ("repo_checks", "repo_checks", "repository compile/config/lint/test checks", (r"\brepository health\b", r"\brepo checks?\b", r"\bcompile/config/lint/test\b", r"\b(?:compile|lint|pytest|tests?).*checks?\b")),
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
    "dependency_audit", "news_search", "web_search", "browse_url", "take_web_screenshot", "geocode_location", "weather_forecast",
    "repo_status", "repo_checks", "page_metadata", "page_links", "extract_document",
    "current_time", "hostname", "environment_summary", "local_subnets", "scan_subnet",
}


_WEATHER_INTENT_RE = re.compile(
    r"\b(?:weather|forecast|current conditions?|precipitation|rainfall|snowfall|humidity|wind speed)\b",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(today|tomorrow|tonight|now|current|this (?:morning|afternoon|evening|week|weekend)|next (?:\d+ )?(?:hours?|days?|week|weekend)|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.I,
)
_LOCATION_STOP = {"the", "weather", "forecast", "today", "tomorrow", "tonight", "now", "current", "please", "like"}


def _clean_entity(value: str) -> str:
    text = re.sub(r"[?!.;,]+$", "", str(value or "").strip())
    text = re.split(r"\b(?:today|tomorrow|tonight|now|this week|next week|next \d+ days?|current)\b", text, maxsplit=1, flags=re.I)[0]
    return re.sub(r"\s+", " ", text).strip(" ,:-")[:180]


def _extract_weather_entity(text: str, previous: dict[str, Any] | None = None) -> str:
    value = " ".join(str(text or "").split())
    # Prefer explicit prepositional location phrases.
    matches = list(re.finditer(r"\b(?:in|at|near)\s+([^?]+)", value, flags=re.I))
    if matches:
        candidate = _clean_entity(matches[-1].group(1))
        if candidate:
            return candidate
    # "weather for Toronto tomorrow" and similar.
    match = re.search(r"\b(?:weather|forecast)\s+(?:for\s+)?(.+)$", value, flags=re.I)
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


def derive_task_frame(user_text: str, previous_frame: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the current task intent/entity/time scope for referential follow-ups."""
    text = " ".join(str(user_text or "").strip().split())
    previous = dict(previous_frame or {})
    intent = ""
    lower = text.lower()
    if _WEATHER_INTENT_RE.search(text):
        intent = "weather"
    elif re.search(r"\b(?:what time is it|current time|current date|what day is it|timezone)\b", lower):
        intent = "current_time"
    elif re.search(r"\b(?:host (?:health|cpu|memory|disk|temperature|state)|system health|host snapshot)\b", lower):
        intent = "host_state"
    elif re.search(r"\b(?:network (?:interfaces|routes|health|state|connections?)|neighbor table|arp table|ndp table)\b", lower):
        intent = "network_state"
    elif re.search(r"\b(?:repo(?:sitory)? (?:status|diff|health)|git status|git diff)\b", lower):
        intent = "repository_state"
    elif re.search(r"(?:\b(?:latest|recent|current|today(?:'s)?)\b.{0,48}\b(?:news|headlines?|stories?)\b|\b(?:latest|top|local)\s+headlines?\b)", lower):
        intent = "news"
    elif is_followup_request(text) and previous.get("intent"):
        intent = str(previous.get("intent"))

    frame: dict[str, Any] = {"intent": intent}
    if intent == "weather":
        frame["entity"] = _extract_weather_entity(text, previous)
        frame["time_scope"] = _extract_time_scope(text, previous) or "current"
    elif intent == "current_time":
        tz = re.search(r"\b(?:in|for)\s+([A-Za-z][A-Za-z0-9_+:/ -]{1,80})$", text)
        if tz:
            frame["entity"] = _clean_entity(tz.group(1))
        elif is_followup_request(text) and str(previous.get("intent") or "") == "current_time":
            frame["entity"] = str(previous.get("entity") or "")
    return {key: value for key, value in frame.items() if value not in {"", None}}


def effective_request_for_frame(user_text: str, frame: dict[str, Any] | None) -> str:
    frame = dict(frame or {})
    if not frame.get("intent"):
        return str(user_text or "")
    parts = [str(frame.get("intent") or "")]
    if frame.get("entity"):
        parts.append(str(frame["entity"]))
    if frame.get("time_scope"):
        parts.append(str(frame["time_scope"]))
    parts.append(str(user_text or ""))
    return " ".join(parts)


def _extract_target(tool: str, text: str) -> str:
    lower = str(text or "")
    if tool == "dns_diagnose":
        match = re.search(r"\bresolve\s+([a-z0-9.-]+)", lower, re.I) or re.search(r"\bdns(?: lookup| resolution| diagnose)?\s+(?:for\s+)?([a-z0-9.-]+)", lower, re.I)
        return str(match.group(1)).lower().rstrip(".") if match else ""
    if tool in {"http_probe", "browse_url"}:
        match = re.search(r"https?://[^\s<>'\"]+", lower, re.I)
        return match.group(0).rstrip(".,)") if match else ""
    if tool == "network_path":
        match = re.search(r"\b(?:traceroute|mtr|network path(?: to)?)\s+([a-z0-9.:-]+)", lower, re.I)
        return str(match.group(1)).lower() if match else ""
    return ""


def _scope_for_requirement(key: str, tool: str, text: str, frame: dict[str, Any]) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    target = _extract_target(tool, text)
    if target:
        scope["target"] = target
    if key.startswith("weather_") or frame.get("intent") == "weather" and tool in {"web_search", "browse_url"}:
        if frame.get("entity"):
            scope["entity"] = frame["entity"]
        if frame.get("time_scope"):
            scope["time_scope"] = frame["time_scope"]
        scope["fact_type"] = "weather"
    return scope


def _scope_text(arguments: Any, result_text: str = "") -> str:
    try:
        import json
        args = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        args = str(arguments or "")
    return (args + " " + str(result_text or "")).lower()


def _scope_matches(scope: dict[str, Any], arguments: Any, result_text: str = "") -> bool:
    if not scope:
        return True
    # Scope metadata is a strengthening of the runtime contract.  Older callers
    # and persisted observations may not provide arguments/result text; treat
    # missing provenance as unknown rather than as a mismatched target.  New
    # execution paths always pass both, so wrong-target successes are rejected.
    if arguments in (None, {}, "") and not str(result_text or "").strip():
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
    if scope.get("time_scope") and isinstance(arguments, dict) and "query" in arguments:
        expected = str(scope["time_scope"]).lower()
        temporal_tokens = [t for t in re.findall(r"[a-z0-9]+", expected) if len(t) > 2]
        if temporal_tokens and not any(token in haystack for token in temporal_tokens):
            return False
    return True


def derive_requirements(user_text: str) -> list[Requirement]:
    """Return ordered, deduplicated requirements explicitly present in a request."""
    text = str(user_text or "")
    lower = text.lower().replace("_", " ")
    result: list[Requirement] = []
    seen_tools: set[str] = set()
    frame = derive_task_frame(text)

    for key, tool, label, patterns in _RULES:
        if any(re.search(pattern, lower, flags=re.I) for pattern in patterns):
            if tool not in seen_tools:
                result.append(Requirement(key=key, tool=tool, label=label, scope=_scope_for_requirement(key, tool, text, frame)))
                seen_tools.add(tool)

    raw_lower = text.lower()
    for tool in sorted(_EXPLICIT_TOOL_NAMES):
        if tool in raw_lower and tool not in seen_tools:
            result.append(Requirement(key=f"explicit:{tool}", tool=tool, label=f"explicitly requested {tool}", scope=_scope_for_requirement(f"explicit:{tool}", tool, text, frame)))
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


def is_followup_request(user_text: str) -> bool:
    """Conservatively detect requests that intentionally depend on the prior task.

    Long, self-contained requests start a fresh task epoch.  Only explicit
    continuity language or short referential follow-ups carry prior task evidence.
    """
    text = " ".join(str(user_text or "").strip().split())
    if not text:
        return False
    lower = text.lower()
    continuity = (
        "without rerunning", "without re-running", "continue", "continue from", "based on that",
        "based on the previous", "previous result", "previous task", "earlier result", "same task",
        "what remains", "what did you find", "those results", "these results", "that result",
        "the above", "follow up", "follow-up",
    )
    if any(phrase in lower for phrase in continuity):
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
        rows = self.requirements
        if pending_only:
            rows = [item for item in rows if item.status not in {"satisfied", "partial"}]
        return [item.tool for item in rows]

    def record_tool(
        self, tool_name: str, *, status: str, reason: str = "", fingerprint: str = "",
        arguments: Any = None, result_text: str = "",
    ) -> None:
        tool_name = str(tool_name or "")
        targets = {tool_name}
        if status in {"ok", "partial"}:
            targets.update(_EQUIVALENT_REQUIREMENT_TOOLS.get(tool_name, ()))
        for item in self.requirements:
            if item.tool not in targets:
                continue
            direct = item.tool == tool_name
            if direct:
                item.attempts += 1
            item.last_reason = str(reason or "")[:120]
            item.fingerprint = str(fingerprint or "")[:32]
            scoped = _scope_matches(item.scope, arguments, result_text)
            if status == "ok" and scoped:
                item.status = "satisfied"
            elif status == "partial" and scoped:
                item.status = "partial"
            elif direct and status in {"ok", "partial"} and not scoped:
                item.status = "failed"
                item.last_reason = "successful tool result did not match requested target/scope"
            elif direct and item.status not in {"satisfied", "partial"}:
                item.status = "failed"

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
