"""Bounded semantic context shared across the main and fast models.

This is prompt-level state sharing, not KV-cache sharing. Different model weights
cannot safely reuse each other's attention cache, but they can consume the same
harness-owned working state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _clip(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if not value or limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    head = max(1, int(limit * 0.62))
    tail = max(1, limit - head - 36)
    return value[:head].rstrip() + "\n...[shared context clipped]...\n" + value[-tail:].lstrip()


def _recent_dialogue(messages: list[dict[str, Any]], max_chars: int) -> str:
    rows: list[str] = []
    # Tool observations belong in the live loop transcript where they are marked
    # untrusted. The bridge shares conversational intent/setup only.
    for message in messages[-10:]:
        role = str(message.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        rows.append(f"{role.upper()}: {content}")
    return _clip("\n".join(rows), max_chars)


def _tool_capabilities(tool_schemas: list[dict[str, Any]], max_chars: int) -> str:
    rows: list[str] = []
    for schema in tool_schemas:
        fn = schema.get("function", {}) if isinstance(schema, dict) else {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        description = " ".join(str(fn.get("description") or "").split())
        params = fn.get("parameters", {}) if isinstance(fn.get("parameters"), dict) else {}
        required = params.get("required", []) if isinstance(params.get("required"), list) else []
        suffix = f" required={','.join(str(x) for x in required)}" if required else ""
        rows.append(f"- {name}: {description[:220]}{suffix}")
    return _clip("\n".join(rows), max_chars)


@dataclass
class SharedModelContext:
    """Harness-owned semantic state visible to both local models.

    The bridge intentionally excludes raw tool output from its base state. Raw
    observations remain in the normal main-model turn and validator transcript,
    where prompt-injection defenses can label them as untrusted data.
    """

    request: str
    summary: str = ""
    relevant_memory: str = ""
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    max_chars: int = 6000
    summary_chars: int = 2200
    recent_chars: int = 1800
    memory_chars: int = 1200
    tool_chars: int = 1600
    validator_events: list[dict[str, Any]] = field(default_factory=list)

    def update_tools(self, tool_schemas: list[dict[str, Any]]) -> None:
        self.tool_schemas = list(tool_schemas or [])

    def add_validator_event(self, report: dict[str, Any], signal: dict[str, Any] | None = None) -> None:
        event = {
            "decision": str(report.get("decision") or "")[:40],
            "diagnosis": str(report.get("diagnosis") or "unknown")[:64],
            "suggested_tool": str(report.get("suggested_tool") or "")[:80],
        }
        if signal:
            event["signal"] = {
                "kind": str(signal.get("kind") or "")[:64],
                "key": str(signal.get("key") or "")[:120],
                "attempts": int(signal.get("attempts") or 0),
            }
        self.validator_events.append(event)
        self.validator_events = self.validator_events[-4:]

    def render(self) -> str:
        sections: list[str] = []
        request = _clip(self.request, 1400)
        if request:
            sections.append("### Current user objective\n" + request)
        summary = _clip(self.summary, self.summary_chars)
        if summary:
            sections.append("### Rolling conversation summary\n" + summary)
        recent = _recent_dialogue(self.recent_messages, self.recent_chars)
        if recent:
            sections.append("### Recent conversational setup\n" + recent)
        memory = _clip(self.relevant_memory, self.memory_chars)
        if memory:
            sections.append("### Relevant recalled context\n" + memory)
        tools = _tool_capabilities(self.tool_schemas, self.tool_chars)
        if tools:
            sections.append("### Currently relevant tool capabilities\n" + tools)
        if self.validator_events:
            sections.append(
                "### Prior fast-model control decisions\n"
                + _clip(json.dumps(self.validator_events, ensure_ascii=False, separators=(",", ":")), 1000)
            )
        return _clip("\n\n".join(sections), max(2000, int(self.max_chars)))
