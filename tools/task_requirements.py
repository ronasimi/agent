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

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tool": self.tool,
            "label": self.label,
            "status": self.status,
            "attempts": self.attempts,
            "last_reason": self.last_reason,
            "fingerprint": self.fingerprint,
        }


_RULES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("host_health", "host_snapshot", "host CPU/memory/disk/temperature state", (r"\bhost (?:health|cpu|memory|disk|temperature)", r"\bcpu,? memory,? disk", r"\btemperature\b")),
    ("pressure", "pressure_snapshot", "CPU/memory/I/O pressure", (r"\bpressure (?:state|snapshot)?\b", r"\b(?:cpu|memory|i/o|io) pressure\b")),
    ("processes", "process_snapshot", "top resource-consuming processes", (r"\btop .*process", r"\bresource[- ]consuming process", r"\bprocess snapshot\b")),
    ("filesystem", "filesystem_snapshot", "filesystem capacity/inode state", (r"\bfilesystem", r"\binode", r"\bdisk capacity\b")),
    ("services", "service_health", "failed/unhealthy service state", (r"\b(?:failed|unhealthy) services?\b", r"\bservice health\b", r"\bservice warnings?\b")),
    ("network_state", "network_snapshot", "network interfaces/routes/listeners", (r"\bnetwork (?:interfaces|routes|health|state)\b", r"\blistening sockets?\b")),
    ("neighbors", "neighbor_snapshot", "network neighbor table", (r"\bneighbors?\b", r"\barp\b", r"\bndp\b")),
    ("connections", "connection_snapshot", "established network connections", (r"\bestablished connections?\b", r"\bconnection snapshot\b", r"\bactive connections?\b")),
    ("dns", "dns_diagnose", "DNS resolution/diagnosis", (r"\bresolve\s+[a-z0-9.-]+", r"\bdns (?:resolution|diagnos|lookup)", r"\bresolve dns\b")),
    ("http", "http_probe", "HTTP/HTTPS connectivity probe", (r"\bprobe (?:https?|connectivity)", r"\bhttps? connectivity\b", r"\bhttp probe\b")),
    ("path", "network_path", "network path/hop diagnosis", (r"\bnetwork path\b", r"\btraceroute\b", r"\bmtr\b", r"\broute tracing\b")),
    ("tool_health", "tool_health", "registered tool/dependency health", (r"\btool/?dependency health\b", r"\btool health\b", r"\bcurrent tool.*health\b")),
    ("dependency_audit", "dependency_audit", "runtime dependency audit", (r"\bdependency health\b", r"\bdependency audit\b", r"\btool/?dependency health\b")),
    ("web_search", "web_search", "current web source discovery", (r"\bweb research\b", r"\bresearch the current\b", r"\bcurrent .*documentation\b", r"\blook up\b", r"\bsearch the web\b")),
    ("web_verify", "browse_url", "authoritative source content verification", (r"\bcurrent .*documentation\b", r"\bofficial .*documentation\b", r"\bsource urls?\b", r"\bverify .*source\b", r"\bweb research\b")),
    ("screenshot", "take_web_screenshot", "requested webpage screenshot", (r"\btake (?:a )?screenshot\b", r"\bscreenshot of\b", r"\bcapture .*page\b")),
    ("repo_status", "repo_status", "repository status", (r"\brepository status\b", r"\brepo status\b")),
    ("repo_checks", "repo_checks", "repository compile/config/lint/test checks", (r"\brepository health\b", r"\brepo checks?\b", r"\bcompile/config/lint/test\b", r"\b(?:compile|lint|pytest|tests?).*checks?\b")),
)

# Explicit tool names in the user's request are requirements as well.  This list
# is intentionally limited to read/diagnostic tools that are meaningful as
# completion checks; arbitrary mutating tools are not auto-required here.
_EXPLICIT_TOOL_NAMES = {
    "host_snapshot", "pressure_snapshot", "process_snapshot", "filesystem_snapshot",
    "service_health", "network_snapshot", "neighbor_snapshot", "connection_snapshot",
    "dns_diagnose", "network_path", "endpoint_probe", "http_probe", "tool_health",
    "dependency_audit", "web_search", "browse_url", "take_web_screenshot",
    "repo_status", "repo_checks", "page_metadata", "page_links", "extract_document",
}


def derive_requirements(user_text: str) -> list[Requirement]:
    """Return ordered, deduplicated requirements explicitly present in a request."""
    text = str(user_text or "")
    lower = text.lower().replace("_", " ")
    result: list[Requirement] = []
    seen_tools: set[str] = set()

    for key, tool, label, patterns in _RULES:
        if any(re.search(pattern, lower, flags=re.I) for pattern in patterns):
            if tool not in seen_tools:
                result.append(Requirement(key=key, tool=tool, label=label))
                seen_tools.add(tool)

    raw_lower = text.lower()
    for tool in sorted(_EXPLICIT_TOOL_NAMES):
        if tool in raw_lower and tool not in seen_tools:
            result.append(Requirement(key=f"explicit:{tool}", tool=tool, label=f"explicitly requested {tool}"))
            seen_tools.add(tool)
    return result


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
    if len(text) <= 180 and re.match(r"^(?:and|also|now|then|so|what about|what else|can you|could you)\b", lower):
        return True
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

    def record_tool(self, tool_name: str, *, status: str, reason: str = "", fingerprint: str = "") -> None:
        for item in self.requirements:
            if item.tool != tool_name:
                continue
            item.attempts += 1
            item.last_reason = str(reason or "")[:120]
            item.fingerprint = str(fingerprint or "")[:32]
            if status == "ok":
                item.status = "satisfied"
            elif status == "partial":
                item.status = "partial"
            elif item.status not in {"satisfied", "partial"}:
                item.status = "failed"

    def pending(self) -> list[Requirement]:
        return [item for item in self.requirements if item.status not in {"satisfied", "partial", "blocked"}]

    def mark_blocked(self, tool_name: str, reason: str) -> None:
        for item in self.requirements:
            if item.tool == tool_name and item.status not in {"satisfied", "partial"}:
                item.status = "blocked"
                item.last_reason = str(reason or "")[:120]

    def as_list(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self.requirements]

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
