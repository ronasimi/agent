"""Persisted harness-owned working state shared by the main and fast models.

The state is operational, task-scoped, and provenance-aware.  It records the
current objective, explicit constraints, completion requirements, provenance-
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
from .conversation_context import DEFAULT_CONVERSATION_ID, get_active_conversation_id, normalize_conversation_id

_STATE_ID = 1
_DEFAULT_LIMITS = {
    "objective_chars": 1600,
    "background_chars": 2600,
    "recent_context_chars": 1600,
    "memory_chars": 1400,
    "evidence_items": 12,
    "evidence_preview_chars": 520,
    "evidence_render_chars": 3600,
    "failure_items": 8,
    "validator_items": 6,
    "plan_items": 6,
    "requirement_items": 24,
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
    # Legacy singleton retained for older installations/tests; new code uses the
    # conversation-keyed table so browser tabs and saved chats cannot share state.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS working_state (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            turn_id INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 2,
            state_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS working_states (
            conversation_id TEXT PRIMARY KEY,
            turn_id INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 3,
            state_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )
        """
    )
    legacy = conn.execute("SELECT turn_id, state_json, updated_at FROM working_state WHERE id=1").fetchone()
    if legacy:
        conn.execute(
            "INSERT OR IGNORE INTO working_states(conversation_id, turn_id, version, state_json, updated_at) VALUES (?, ?, 3, ?, ?)",
            (DEFAULT_CONVERSATION_ID, int(legacy[0] or 0), str(legacy[1] or '{}'), str(legacy[2] or utc_now())),
        )
    return conn


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "turn_id": 0,
        "task_epoch": 0,
        "task_frame": {},
        "status": "idle",
        "objective": "",
        "background": {"rolling_summary": "", "recent_context": "", "recalled_context": ""},
        "constraints": [],
        "requirements": [],
        "tool_capabilities": [],
        "verified_observations": [],
        "failed_approaches": [],
        "open_questions": [],
        "current_plan": [],
        "validator_history": [],
        "updated_at": utc_now(),
    }


def _resolved_conversation_id(conversation_id: str | None = None) -> str:
    return normalize_conversation_id(conversation_id or get_active_conversation_id())


def _load(conversation_id: str | None = None) -> dict[str, Any]:
    cid = _resolved_conversation_id(conversation_id)
    with _connect() as conn:
        row = conn.execute("SELECT state_json FROM working_states WHERE conversation_id = ?", (cid,)).fetchone()
    if not row:
        return _empty_state()
    try:
        value = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return _empty_state()
    if not isinstance(value, dict):
        return _empty_state()
    merged = _empty_state()
    merged.update(value)
    merged.setdefault("task_epoch", 0)
    merged.setdefault("task_frame", {})
    merged.setdefault("requirements", [])
    return merged


def _save(state: dict[str, Any], conversation_id: str | None = None) -> None:
    cid = _resolved_conversation_id(conversation_id)
    state = dict(state or {})
    state["schema_version"] = 3
    state["updated_at"] = utc_now()
    turn_id = int(state.get("turn_id") or 0)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO working_states(conversation_id, turn_id, version, state_json, updated_at)
            VALUES(?, ?, 3, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                turn_id=excluded.turn_id,
                version=excluded.version,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at
            """,
            (cid, turn_id, _json(state), state["updated_at"]),
        )
        if cid == DEFAULT_CONVERSATION_ID:
            conn.execute(
                """INSERT INTO working_state(id, turn_id, version, state_json, updated_at)
                   VALUES(1, ?, 3, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET turn_id=excluded.turn_id, version=excluded.version,
                     state_json=excluded.state_json, updated_at=excluded.updated_at""",
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


def _tool_capabilities(tool_schemas: list[dict[str, Any]], max_items: int = 28) -> list[dict[str, Any]]:
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


def _clean_requirements(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for item in list(items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        clean.append({
            "key": _clip(item.get("key"), 80),
            "tool": _clip(item.get("tool"), 80),
            "label": _clip(item.get("label"), 180),
            "status": _clip(item.get("status") or "pending", 24),
            "attempts": int(item.get("attempts") or 0),
            "last_reason": _clip(item.get("last_reason"), 100),
            "fingerprint": _clip(item.get("fingerprint"), 32),
            "scope": item.get("scope") if isinstance(item.get("scope"), dict) else {},
        })
    return clean


@dataclass
class WorkingStateStore:
    """Small persisted source-of-truth object owned exclusively by the harness."""

    limits: dict[str, int] | None = None
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        merged = dict(_DEFAULT_LIMITS)
        for key, value in (self.limits or {}).items():
            if key in merged:
                try:
                    merged[key] = max(1, int(value))
                except (TypeError, ValueError):
                    pass
        self.limits = merged
        if self.conversation_id is not None:
            self.conversation_id = _resolved_conversation_id(self.conversation_id)
        with _connect():
            pass

    def _cid(self) -> str:
        return _resolved_conversation_id(self.conversation_id)

    def load(self) -> dict[str, Any]:
        return _load(self._cid())

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
        requirements: list[dict[str, Any]] | None = None,
        continuation: bool = False,
        task_frame: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = _load(self._cid())
        state = _empty_state()
        previous_epoch = int(previous.get("task_epoch") or 0)
        state["task_epoch"] = previous_epoch if continuation and previous_epoch else previous_epoch + 1
        if state["task_epoch"] <= 0:
            state["task_epoch"] = 1

        carried_constraints: list[str] = []
        if continuation:
            # Follow-ups may rely on verified evidence and failed approaches from
            # the immediately preceding task. New top-level requests never do.
            state["verified_observations"] = list(previous.get("verified_observations") or [])[-self.limits["evidence_items"]:]
            state["failed_approaches"] = list(previous.get("failed_approaches") or [])[-self.limits["failure_items"]:]
            state["validator_history"] = list(previous.get("validator_history") or [])[-self.limits["validator_items"]:]
            carried_constraints = list(previous.get("constraints") or [])[-4:]

        current_constraints = _extract_constraints(objective, policy_note)
        constraints: list[str] = []
        for item in [*carried_constraints, *current_constraints]:
            if item and item not in constraints:
                constraints.append(item)

        current_requirements = _clean_requirements(requirements or [], self.limits["requirement_items"])
        if continuation:
            previous_requirements = _clean_requirements(list(previous.get("requirements") or []), self.limits["requirement_items"])
            merged_requirements: list[dict[str, Any]] = []
            seen_keys: set[tuple[str, str]] = set()
            for item in [*previous_requirements, *current_requirements]:
                marker = (str(item.get("key") or ""), str(item.get("tool") or ""))
                if marker in seen_keys:
                    # Current-turn requirement state wins when the same requirement
                    # is explicitly requested again.
                    for index, existing in enumerate(merged_requirements):
                        if (str(existing.get("key") or ""), str(existing.get("tool") or "")) == marker:
                            merged_requirements[index] = item
                            break
                else:
                    merged_requirements.append(item)
                    seen_keys.add(marker)
            state_requirements = merged_requirements[-self.limits["requirement_items"]:]
        else:
            state_requirements = current_requirements

        state.update({
            "turn_id": max(0, int(turn_id)),
            "status": "active",
            "objective": _clip(objective, self.limits["objective_chars"]),
            "task_frame": dict(task_frame or {}),
            "background": {
                "rolling_summary": _clip(rolling_summary, self.limits["background_chars"]) if continuation else "",
                # Avoid stale task leakage. Raw recent conversational setup is
                # only carried when the current request explicitly refers back.
                "recent_context": _recent_context(recent_messages, self.limits["recent_context_chars"]) if continuation else "",
                "recalled_context": _clip(recalled_context, self.limits["memory_chars"]),
            },
            "constraints": constraints[:8],
            "requirements": state_requirements,
            "tool_capabilities": _tool_capabilities(tool_schemas),
        })
        _save(state, self._cid())
        return state

    def update_tools(self, tool_schemas: list[dict[str, Any]]) -> None:
        state = _load(self._cid())
        state["tool_capabilities"] = _tool_capabilities(tool_schemas)
        _save(state, self._cid())

    def update_requirements(self, requirements: list[dict[str, Any]]) -> None:
        state = _load(self._cid())
        state["requirements"] = _clean_requirements(requirements, self.limits["requirement_items"])
        _save(state, self._cid())

    def set_plan(self, plan: list[dict[str, Any]] | list[str]) -> None:
        state = _load(self._cid())
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
        _save(state, self._cid())

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
        state = _load(self._cid())
        status = str(status or "error")
        from .grounding import grounding_metadata
        grounding = grounding_metadata(tool_name, result_text, arguments=arguments)
        record = {
            "tool": _clip(tool_name, 80),
            "status": status,
            "reason": _clip(reason, 100),
            "arguments_digest": _args_digest(arguments),
            "arguments": grounding.get("arguments", {}),
            "target": grounding.get("target", ""),
            "time_scope": grounding.get("time_scope", ""),
            "source_url": grounding.get("source_url", ""),
            "discovered_urls": grounding.get("discovered_urls", []),
            "fingerprint": _clip(fingerprint, 32),
            "evidence_ref": _clip(observation_id, 64),
            # Persisted for an explicitly untrusted evidence digest. It is never
            # inserted into the system-role canonical metadata block.
            "evidence_preview": _clip(_normalize_space(result_text), self.limits["evidence_preview_chars"]),
            "fact_types": grounding["fact_types"],
            "source_tools": grounding["source_tools"],
            "weather_verified": bool(grounding["weather_verified"]),
            "turn_id": int(state.get("turn_id") or 0),
            "at": utc_now(),
        }
        if status in {"ok", "partial"}:
            observations = list(state.get("verified_observations") or [])
            # Deduplicate exact repeated observations to keep ingestion stable.
            duplicate = next((item for item in observations if item.get("tool") == record["tool"] and item.get("fingerprint") == record["fingerprint"] and record["fingerprint"]), None)
            if duplicate is None:
                observations.append(record)
            else:
                duplicate.update(record)
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
        state["current_plan"] = []
        _save(state, self._cid())

    def record_validator(self, report: dict[str, Any], signal: dict[str, Any] | None = None) -> None:
        state = _load(self._cid())
        event: dict[str, Any] = {
            "decision": _clip(report.get("decision"), 40),
            "diagnosis": _clip(report.get("diagnosis") or "unknown", 64),
            "suggested_tool": _clip(report.get("suggested_tool"), 80),
            "suggested_recipe": _clip(report.get("name") if report.get("decision") == "recipe" else "", 80),
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
        _save(state, self._cid())

    def complete_turn(self, *, blocked: bool = False) -> None:
        state = _load(self._cid())
        state["status"] = "blocked" if blocked else "complete"
        state["current_plan"] = []
        _save(state, self._cid())

    def render_evidence(self, max_chars: int | None = None) -> str:
        """Render bounded untrusted evidence excerpts for a user-role prompt block."""
        state = _load(self._cid())
        limit = max(400, int(max_chars or self.limits["evidence_render_chars"]))
        rows: list[dict[str, Any]] = []
        for item in list(state.get("verified_observations") or [])[-self.limits["evidence_items"]:]:
            if not isinstance(item, dict):
                continue
            rows.append({
                "tool": item.get("tool", ""),
                "status": item.get("status", ""),
                "reason": item.get("reason", ""),
                "evidence_ref": item.get("evidence_ref", ""),
                "target": item.get("target", ""),
                "time_scope": item.get("time_scope", ""),
                "source_url": item.get("source_url", ""),
                "excerpt": _clip(item.get("evidence_preview"), min(420, self.limits["evidence_preview_chars"])),
            })
        text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= limit:
            return text
        # Drop oldest entries first; never raw-slice JSON.
        while len(rows) > 1 and len(text) > limit:
            rows.pop(0)
            text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= limit:
            return text
        if rows:
            excerpt_limit = max(40, limit // 3)
            while excerpt_limit >= 40:
                rows[0]["excerpt"] = _clip(rows[0].get("excerpt"), excerpt_limit)
                text = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
                if len(text) <= limit:
                    return text
                excerpt_limit //= 2
        return "[]"

    def render(self, *, include_tool_capabilities: bool = True) -> str:
        """Render valid bounded canonical metadata JSON.

        The main model already receives native tool schemas, so callers may omit
        the duplicate capability descriptions. The fast validator can retain them.
        """
        state = _load(self._cid())
        compact = {
            "turn_id": state.get("turn_id", 0),
            "task_epoch": state.get("task_epoch", 0),
            "task_frame": dict(state.get("task_frame", {}) or {}),
            "status": state.get("status", "idle"),
            "objective": state.get("objective", ""),
            "background": dict(state.get("background", {}) or {}),
            "constraints": list(state.get("constraints", []) or []),
            "requirements": _clean_requirements(list(state.get("requirements", []) or []), self.limits["requirement_items"]),
            "tool_capabilities": list(state.get("tool_capabilities", []) or []) if include_tool_capabilities else [],
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

        bg = compact["background"]
        bg["rolling_summary"] = _clip(bg.get("rolling_summary"), 700)
        bg["recent_context"] = _clip(bg.get("recent_context"), 500)
        bg["recalled_context"] = _clip(bg.get("recalled_context"), 400)
        compact["objective"] = _clip(compact.get("objective"), 800)
        for item in compact["tool_capabilities"]:
            if isinstance(item, dict):
                item["description"] = _clip(item.get("description"), 90)
        text = dump()

        reducers = [
            ("tool_capabilities", 8),
            ("verified_observations", 5),
            ("failed_approaches", 4),
            ("validator_history", 3),
            ("constraints", 4),
            ("requirements", 8),
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
                "rolling_summary": _clip(bg.get("rolling_summary"), 360),
                "recent_context": _clip(bg.get("recent_context"), 240),
                "recalled_context": _clip(bg.get("recalled_context"), 200),
            }
            compact["objective"] = _clip(compact.get("objective"), 520)
            text = dump()

        if len(text) > limit:
            minimal = {
                "turn_id": compact.get("turn_id", 0),
                "task_epoch": compact.get("task_epoch", 0),
                "status": compact.get("status", "idle"),
                "objective": _clip(compact.get("objective"), 360),
                "constraints": compact.get("constraints", [])[-3:],
                "requirements": compact.get("requirements", [])[-8:],
                "verified_observations": compact.get("verified_observations", [])[-3:],
                "failed_approaches": compact.get("failed_approaches", [])[-3:],
                "current_plan": compact.get("current_plan", [])[-3:],
                "validator_history": compact.get("validator_history", [])[-2:],
            }
            text = json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))
        return text if len(text) <= limit else json.dumps({
            "turn_id": compact.get("turn_id", 0),
            "task_epoch": compact.get("task_epoch", 0),
            "status": compact.get("status", "idle"),
            "objective": _clip(compact.get("objective"), max(80, limit // 3)),
        }, ensure_ascii=False, separators=(",", ":"))

    def clear(self) -> None:
        with _connect() as conn:
            conn.execute("DELETE FROM working_states WHERE conversation_id = ?", (self._cid(),))
            if self._cid() == DEFAULT_CONVERSATION_ID:
                conn.execute("DELETE FROM working_state WHERE id = ?", (_STATE_ID,))
