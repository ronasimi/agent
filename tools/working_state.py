"""Persisted harness-owned working state shared by the main and fast models.

The state is deliberately operational rather than a second conversational
memory.  It records the current objective, explicit constraints, provenance-
tagged tool evidence, failed approaches, the current plan, and structured
validator decisions.  Only harness code commits updates.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from .runtime import DB_PATH, DB_TIMEOUT, utc_now

_STATE_ID = 1
_DEFAULT_LIMITS = {
    "objective_chars": 1600,
    "background_chars": 2600,
    "recent_context_chars": 1600,
    "memory_chars": 1400,
    "evidence_items": 10,
    "evidence_preview_chars": 520,
    "failure_items": 8,
    "validator_items": 6,
    "plan_items": 6,
    "max_render_chars": 7000,
}


def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    limit = max(0, int(limit))
    if not text or not limit:
        return ""
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.65))
    tail = max(1, limit - head - 29)
    return text[:head].rstrip() + " …[clipped]… " + text[-tail:].lstrip()


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS working_state (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            turn_id INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            state_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "turn_id": 0,
        "status": "idle",
        "objective": "",
        "background": {"rolling_summary": "", "recent_context": "", "recalled_context": ""},
        "constraints": [],
        "tool_capabilities": [],
        "verified_observations": [],
        "failed_approaches": [],
        "open_questions": [],
        "current_plan": [],
        "validator_history": [],
        "updated_at": utc_now(),
    }


def _load() -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute("SELECT state_json FROM working_state WHERE id = ?", (_STATE_ID,)).fetchone()
    if not row:
        return _empty_state()
    try:
        value = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return _empty_state()
    return value if isinstance(value, dict) else _empty_state()


def _save(state: dict[str, Any]) -> None:
    state = dict(state or {})
    state["updated_at"] = utc_now()
    turn_id = int(state.get("turn_id") or 0)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO working_state(id, turn_id, version, state_json, updated_at)
            VALUES(1, ?, 1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                turn_id=excluded.turn_id,
                version=excluded.version,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at
            """,
            (turn_id, _json(state), state["updated_at"]),
        )


def _extract_constraints(user_text: str, policy_note: str, limit: int = 8) -> list[str]:
    items: list[str] = []
    note = _normalize_space(policy_note)
    if note:
        items.append(note)
    for raw in str(user_text or "").splitlines():
        line = _normalize_space(raw).lstrip("-*•0123456789. )")
        if not line:
            continue
        lower = line.lower()
        if re.search(r"\b(do not|don't|never|must|only|unless|without|keep .*read[- ]only)\b", lower):
            clipped = _clip(line, 300)
            if clipped and clipped not in items:
                items.append(clipped)
        if len(items) >= limit:
            break
    return items[:limit]


def _tool_capabilities(tool_schemas: list[dict[str, Any]], max_items: int = 24) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for schema in tool_schemas[: max(1, int(max_items))]:
        fn = schema.get("function", {}) if isinstance(schema, dict) else {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        params = fn.get("parameters", {}) if isinstance(fn.get("parameters"), dict) else {}
        required = params.get("required", []) if isinstance(params.get("required"), list) else []
        result.append({
            "name": name,
            "required": [str(item) for item in required[:8]],
            "description": _clip(_normalize_space(fn.get("description")), 180),
        })
    return result


def _recent_context(messages: list[dict[str, Any]], max_chars: int) -> str:
    rows: list[str] = []
    for message in messages[-8:]:
        role = str(message.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        content = _normalize_space(message.get("content"))
        if not content:
            continue
        rows.append(f"{role.upper()}: {content}")
    return _clip("\n".join(rows), max_chars)


def _args_digest(arguments: Any) -> str:
    try:
        text = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(arguments)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


@dataclass
class WorkingStateStore:
    """Small persisted source-of-truth object owned exclusively by the harness."""

    limits: dict[str, int] | None = None

    def __post_init__(self) -> None:
        merged = dict(_DEFAULT_LIMITS)
        for key, value in (self.limits or {}).items():
            if key in merged:
                try:
                    merged[key] = max(1, int(value))
                except (TypeError, ValueError):
                    pass
        self.limits = merged
        # Ensure the table exists eagerly so restart behavior is deterministic.
        with _connect():
            pass

    def load(self) -> dict[str, Any]:
        return _load()

    def begin_turn(
        self,
        *,
        turn_id: int,
        objective: str,
        rolling_summary: str,
        recalled_context: str,
        recent_messages: list[dict[str, Any]],
        policy_note: str,
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        state = _empty_state()
        state.update({
            "turn_id": max(0, int(turn_id)),
            "status": "active",
            "objective": _clip(objective, self.limits["objective_chars"]),
            "background": {
                "rolling_summary": _clip(rolling_summary, self.limits["background_chars"]),
                "recent_context": _recent_context(recent_messages, self.limits["recent_context_chars"]),
                "recalled_context": _clip(recalled_context, self.limits["memory_chars"]),
            },
            "constraints": _extract_constraints(objective, policy_note),
            "tool_capabilities": _tool_capabilities(tool_schemas),
        })
        _save(state)
        return state

    def update_tools(self, tool_schemas: list[dict[str, Any]]) -> None:
        state = _load()
        state["tool_capabilities"] = _tool_capabilities(tool_schemas)
        _save(state)

    def set_plan(self, plan: list[dict[str, Any]] | list[str]) -> None:
        state = _load()
        clean: list[Any] = []
        for item in list(plan or [])[: self.limits["plan_items"]]:
            if isinstance(item, dict):
                clean.append({
                    "action": _clip(item.get("action"), 80),
                    "tool": _clip(item.get("tool"), 80),
                    "arguments_digest": _clip(item.get("arguments_digest"), 24),
                })
            else:
                clean.append(_clip(item, 240))
        state["current_plan"] = clean
        _save(state)

    def record_tool_result(
        self,
        *,
        tool_name: str,
        arguments: Any,
        status: str,
        reason: str,
        result_text: str,
        fingerprint: str = "",
        observation_id: str = "",
    ) -> None:
        state = _load()
        status = str(status or "error")
        record = {
            "tool": _clip(tool_name, 80),
            "status": status,
            "reason": _clip(reason, 100),
            "arguments_digest": _args_digest(arguments),
            "fingerprint": _clip(fingerprint, 32),
            "evidence_ref": _clip(observation_id, 64),
            # Tool output remains data, never policy. The raw/full value stays in
            # the normal loop or tool_observations table. This bounded preview is
            # persisted for diagnostics but deliberately omitted from render().
            "evidence_preview": _clip(_normalize_space(result_text), self.limits["evidence_preview_chars"]),
            "at": utc_now(),
        }
        if status in {"ok", "partial"}:
            observations = list(state.get("verified_observations") or [])
            observations.append(record)
            state["verified_observations"] = observations[-self.limits["evidence_items"]:]
        else:
            failures = list(state.get("failed_approaches") or [])
            failures.append({
                "tool": record["tool"],
                "reason": record["reason"],
                "arguments_digest": record["arguments_digest"],
                "at": record["at"],
            })
            state["failed_approaches"] = failures[-self.limits["failure_items"]:]
        # A completed tool call consumes the immediate plan item. The next model
        # step can establish a new one.
        state["current_plan"] = []
        _save(state)

    def record_validator(self, report: dict[str, Any], signal: dict[str, Any] | None = None) -> None:
        state = _load()
        event: dict[str, Any] = {
            "decision": _clip(report.get("decision"), 40),
            "diagnosis": _clip(report.get("diagnosis") or "unknown", 64),
            "suggested_tool": _clip(report.get("suggested_tool"), 80),
            "at": utc_now(),
        }
        if signal:
            event["signal"] = {
                "kind": _clip(signal.get("kind"), 64),
                "key": _clip(signal.get("key"), 100),
                "attempts": int(signal.get("attempts") or 0),
            }
        history = list(state.get("validator_history") or [])
        history.append(event)
        state["validator_history"] = history[-self.limits["validator_items"]:]
        suggested = event.get("suggested_tool")
        if suggested and event.get("decision") in {"retry", "switch_tool", "corrective_tool"}:
            state["current_plan"] = [{"action": "validator_recovery", "tool": suggested, "arguments_digest": ""}]
        if event.get("decision") == "blocked":
            questions = list(state.get("open_questions") or [])
            questions.append(_clip(f"Blocked after validator diagnosis: {event.get('diagnosis')}", 240))
            state["open_questions"] = questions[-4:]
        _save(state)

    def complete_turn(self, *, blocked: bool = False) -> None:
        state = _load()
        state["status"] = "blocked" if blocked else "complete"
        state["current_plan"] = []
        _save(state)

    def render(self) -> str:
        """Render valid bounded JSON, trimming low-priority detail structurally.

        The returned string stays parseable even when the state is large; this
        matters because both models are told that the block is harness-owned JSON.
        """
        state = _load()
        compact = {
            "turn_id": state.get("turn_id", 0),
            "status": state.get("status", "idle"),
            "objective": state.get("objective", ""),
            "background": dict(state.get("background", {}) or {}),
            "constraints": list(state.get("constraints", []) or []),
            "tool_capabilities": list(state.get("tool_capabilities", []) or []),
            # Never promote raw tool/web text into the system-role state block.
            # The live tool transcript carries content as explicitly untrusted
            # data; this canonical index carries only provenance/status metadata.
            "verified_observations": [
                {key: value for key, value in item.items() if key != "evidence_preview"}
                for item in list(state.get("verified_observations", []) or [])
                if isinstance(item, dict)
            ],
            "failed_approaches": list(state.get("failed_approaches", []) or []),
            "open_questions": list(state.get("open_questions", []) or []),
            "current_plan": list(state.get("current_plan", []) or []),
            "validator_history": list(state.get("validator_history", []) or []),
        }
        limit = self.limits["max_render_chars"]

        def dump() -> str:
            return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

        text = dump()
        if len(text) <= limit:
            return text

        # First shrink verbose strings while keeping every category present.
        bg = compact["background"]
        bg["rolling_summary"] = _clip(bg.get("rolling_summary"), 900)
        bg["recent_context"] = _clip(bg.get("recent_context"), 700)
        bg["recalled_context"] = _clip(bg.get("recalled_context"), 500)
        compact["objective"] = _clip(compact.get("objective"), 900)
        for item in compact["tool_capabilities"]:
            if isinstance(item, dict):
                item["description"] = _clip(item.get("description"), 100)
        for item in compact["verified_observations"]:
            if isinstance(item, dict):
                item["evidence_preview"] = _clip(item.get("evidence_preview"), 220)
        text = dump()

        # Then discard oldest/redundant detail in a deterministic priority order.
        reducers = [
            ("tool_capabilities", 8),
            ("verified_observations", 5),
            ("failed_approaches", 4),
            ("validator_history", 3),
            ("constraints", 4),
        ]
        for key, floor in reducers:
            values = compact.get(key, [])
            while len(text) > limit and isinstance(values, list) and len(values) > floor:
                if key == "tool_capabilities":
                    values.pop()
                else:
                    values.pop(0)
                text = dump()

        if len(text) > limit:
            compact["background"] = {
                "rolling_summary": _clip(bg.get("rolling_summary"), 420),
                "recent_context": _clip(bg.get("recent_context"), 360),
                "recalled_context": _clip(bg.get("recalled_context"), 260),
            }
            compact["objective"] = _clip(compact.get("objective"), 600)
            for item in compact["verified_observations"]:
                if isinstance(item, dict):
                    item["evidence_preview"] = _clip(item.get("evidence_preview"), 120)
            text = dump()

        if len(text) > limit:
            # Last-resort minimal state remains valid JSON and preserves the
            # highest-value control information.
            minimal = {
                "turn_id": compact.get("turn_id", 0),
                "status": compact.get("status", "idle"),
                "objective": _clip(compact.get("objective"), 400),
                "constraints": compact.get("constraints", [])[-3:],
                "verified_observations": compact.get("verified_observations", [])[-3:],
                "failed_approaches": compact.get("failed_approaches", [])[-3:],
                "current_plan": compact.get("current_plan", [])[-3:],
                "validator_history": compact.get("validator_history", [])[-2:],
            }
            text = json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))
        return text if len(text) <= limit else json.dumps({
            "turn_id": compact.get("turn_id", 0),
            "status": compact.get("status", "idle"),
            "objective": _clip(compact.get("objective"), max(80, limit // 3)),
        }, ensure_ascii=False, separators=(",", ":"))

    def clear(self) -> None:
        with _connect() as conn:
            conn.execute("DELETE FROM working_state WHERE id = ?", (_STATE_ID,))
