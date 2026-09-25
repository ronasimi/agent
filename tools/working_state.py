"""Persisted harness-owned working state shared by the main and fast models.

The state is operational, task-scoped, and provenance-aware.  It records the
current objective, explicit constraints, completion requirements, provenance-
tagged tool evidence, failed approaches, the current plan, and structured
validator decisions.  Only harness code commits updates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .runtime import DB_PATH, DB_TIMEOUT, utc_now
from .conversation_context import DEFAULT_CONVERSATION_ID, get_active_conversation_id, normalize_conversation_id

_STATE_ID = 1
# Tier-1 working state is read constantly during a tool loop. Keep the active
# JSON object in process RAM and write through to SQLite/WAL for crash recovery.
_STATE_CACHE_LOCK = threading.RLock()
_STATE_CACHE: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
_STATE_CACHE_MAX = 64
_STATE_SCHEMA_LOCK = threading.RLock()
_INITIALIZED_STATE_DBS: set[str] = set()

def _state_cache_key(conversation_id: str) -> tuple[str, str]:
    return (str(DB_PATH), str(conversation_id))

def _cache_state(conversation_id: str, state: dict[str, Any]) -> None:
    key = _state_cache_key(conversation_id)
    with _STATE_CACHE_LOCK:
        _STATE_CACHE[key] = copy.deepcopy(state)
        _STATE_CACHE.move_to_end(key)
        while len(_STATE_CACHE) > _STATE_CACHE_MAX:
            _STATE_CACHE.popitem(last=False)

def _cached_state(conversation_id: str) -> dict[str, Any] | None:
    key = _state_cache_key(conversation_id)
    with _STATE_CACHE_LOCK:
        value = _STATE_CACHE.get(key)
        if value is None:
            return None
        _STATE_CACHE.move_to_end(key)
        return copy.deepcopy(value)

def _invalidate_state_cache(conversation_id: str) -> None:
    with _STATE_CACHE_LOCK:
        _STATE_CACHE.pop(_state_cache_key(conversation_id), None)

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
    # Large explicitly enumerated audit/stress-test requests may contain dozens
    # of independent requirements. Keep the durable scheduler large enough to
    # preserve them all while render() continues to expose only the active step.
    "scheduler_steps": 96,
    "scheduler_task_chars": 1200,
    "scheduler_result_chars": 900,
    "scheduler_objective_chars": 24000,
    # Persist the complete operational ledger separately from the smaller
    # prompt-facing requirement window. Large deterministic plans (for example
    # 37-item tool/recipe audits) must remain fully inspectable/resumable without
    # paying their full token cost on every model call.
    "requirement_store_items": 96,
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


def _configure_connection(conn: sqlite3.Connection, *, initialize: bool = False) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA synchronous=NORMAL")
    if initialize:
        # WAL is persistent database configuration. Do this once during schema
        # setup rather than on every hot-path state read/write connection.
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_schema() -> None:
    """Initialize Tier-1 durable tables once per database path.

    Working-state reads happen repeatedly inside a single tool loop. The active
    state itself is served from RAM; when SQLite is needed, avoid paying DDL and
    migration checks on every connection.
    """
    db = str(DB_PATH)
    with _STATE_SCHEMA_LOCK:
        if db in _INITIALIZED_STATE_DBS:
            return
        conn = _configure_connection(sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT), initialize=True)
        try:
            # Legacy singleton retained for older installations/tests; new code
            # uses the conversation-keyed table so saved chats cannot share state.
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
            conn.commit()
            _INITIALIZED_STATE_DBS.add(db)
        finally:
            conn.close()


def _connect() -> sqlite3.Connection:
    _ensure_schema()
    return _configure_connection(sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT))


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "turn_id": 0,
        "task_epoch": 0,
        "task_frame": {},
        "fact_frames": {},
        "fact_requirements": [],
        "status": "idle",
        "objective": "",
        "persistent_goal": {},
        "background": {"rolling_summary": "", "recent_context": "", "recalled_context": ""},
        "constraints": [],
        "requirements": [],
        "tool_capabilities": [],
        "verified_observations": [],
        "failed_approaches": [],
        "open_questions": [],
        "current_plan": [],
        "scheduler": {
            "enabled": False,
            "overall_objective": "",
            "active_index": 0,
            "steps": [],
        },
        "validator_history": [],
        "updated_at": utc_now(),
    }


def _resolved_conversation_id(conversation_id: str | None = None) -> str:
    return normalize_conversation_id(conversation_id or get_active_conversation_id())


def _load(conversation_id: str | None = None) -> dict[str, Any]:
    cid = _resolved_conversation_id(conversation_id)
    cached = _cached_state(cid)
    if cached is not None:
        return cached
    with _connect() as conn:
        row = conn.execute("SELECT state_json FROM working_states WHERE conversation_id = ?", (cid,)).fetchone()
    if not row:
        value = _empty_state()
        _cache_state(cid, value)
        return copy.deepcopy(value)
    try:
        value = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        value = _empty_state()
    if not isinstance(value, dict):
        value = _empty_state()
    merged = _empty_state()
    merged.update(value)
    merged.setdefault("task_epoch", 0)
    merged.setdefault("task_frame", {})
    merged.setdefault("fact_frames", {})
    merged.setdefault("fact_requirements", [])
    merged.setdefault("requirements", [])
    merged.setdefault("persistent_goal", {})
    merged.setdefault("scheduler", {
        "enabled": False,
        "overall_objective": "",
        "active_index": 0,
        "steps": [],
    })
    _cache_state(cid, merged)
    return copy.deepcopy(merged)

def _save(state: dict[str, Any], conversation_id: str | None = None) -> None:
    cid = _resolved_conversation_id(conversation_id)
    state = dict(state or {})
    state["schema_version"] = 3
    state["updated_at"] = utc_now()
    turn_id = int(state.get("turn_id") or 0)
    state_json = _json(state)
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
            (cid, turn_id, state_json, state["updated_at"]),
        )
        if cid == DEFAULT_CONVERSATION_ID:
            conn.execute(
                """INSERT INTO working_state(id, turn_id, version, state_json, updated_at)
                   VALUES(1, ?, 3, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET turn_id=excluded.turn_id, version=excluded.version,
                     state_json=excluded.state_json, updated_at=excluded.updated_at""",
                (turn_id, state_json, state["updated_at"]),
            )
    _cache_state(cid, state)


def _extract_constraints(user_text: str, policy_note: str, limit: int = 32) -> list[str]:
    items: list[str] = []
    note = _normalize_space(policy_note)
    if note:
        items.append(note)
    in_explicit_constraint_section = False
    for raw in str(user_text or "").splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", raw)
        if heading:
            title = _normalize_space(heading.group(1)).casefold()
            in_explicit_constraint_section = bool(
                re.search(r"\b(?:safety|constraint|guardrail)\b", title)
            )
            continue
        line = _normalize_space(raw).lstrip("-*•0123456789. )")
        if not line:
            continue
        lower = line.lower()
        if in_explicit_constraint_section or re.search(
            r"\b(do not|don't|never|must|only|unless|without|prefer|keep .*read[- ]only|"
            r"use .*only when|verify important claims|complete the entire)\b",
            lower,
        ):
            clipped = _clip(line, 300)
            if clipped and clipped not in items:
                items.append(clipped)
        if len(items) >= limit:
            break
    return items[:limit]


def _structured_plan_global_constraint_text(user_text: str) -> str:
    """Return only the plan preamble/global-constraint portion of a suite.

    Structured plans persist each numbered requirement separately.  Re-running
    the generic constraint extractor over the entire original suite used to
    promote step-local prose such as "Do not infer missing hardware details" or
    "without executing it yet" into global model-facing constraints.  Besides
    leaking future requirements, those strings accumulated prompt cost on every
    scheduler step.  Global policy lives before the first numbered requirement,
    so keep only that preamble here.
    """
    text = str(user_text or "")
    match = re.search(r"(?m)^\s*#{1,6}\s+\d+\.\s+", text)
    if match:
        return text[: match.start()]
    # Fallback for explicit numbered suites without Markdown headings.
    match = re.search(r"(?m)^\s*\d+\.\s+[A-Z][^\n]{0,120}$", text)
    return text[: match.start()] if match else text


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


def _clean_requirements(
    items: list[dict[str, Any]], limit: int, *, evidence_preview_chars: int = 280,
) -> list[dict[str, Any]]:
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
            "evidence": [
                {
                    "source": _clip(row.get("source"), 32),
                    "tool": _clip(row.get("tool"), 80),
                    "status": _clip(row.get("status"), 24),
                    "reason": _clip(row.get("reason"), 100),
                    "fingerprint": _clip(row.get("fingerprint"), 32),
                    "arguments_digest": _clip(row.get("arguments_digest"), 24),
                    "evidence_ref": _clip(row.get("evidence_ref"), 80),
                    **({"evidence_preview": _clip(row.get("evidence_preview"), evidence_preview_chars)}
                       if evidence_preview_chars > 0 and row.get("evidence_preview") else {}),
                }
                for row in list(item.get("evidence") or [])[-4:] if isinstance(row, dict)
            ],
        })
    return clean

def _bounded_observations(state: dict[str, Any], observations: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Bound evidence while pinning proof for satisfied requirements.

    A compound turn may generate many sibling observations. Dropping the only
    observation that satisfied an already-closed requirement forces the model
    to re-fetch work the ledger still considers complete. Keep the newest proof
    row for every satisfied/partial requirement tool (and fact-ledger evidence
    tool) before filling the remaining slots with the newest observations.
    """
    limit = max(1, int(limit))
    rows = [dict(item) for item in observations if isinstance(item, dict)]
    if len(rows) <= limit:
        return rows

    pinned_tools: set[str] = set()
    for item in list(state.get("requirements") or []):
        if not isinstance(item, dict) or str(item.get("status") or "") not in {"satisfied", "partial"}:
            continue
        tool = str(item.get("tool") or "").strip()
        if tool:
            pinned_tools.add(tool)
    for item in list(state.get("fact_requirements") or []):
        if not isinstance(item, dict) or not bool(item.get("satisfied")):
            continue
        for tool in item.get("evidence") or []:
            value = str(tool or "").strip()
            if value.startswith("stored:"):
                value = value.split(":", 1)[1].strip()
            if value:
                pinned_tools.add(value)

    pinned_indexes: set[int] = set()
    for tool in pinned_tools:
        for index in range(len(rows) - 1, -1, -1):
            if str(rows[index].get("tool") or "") == tool:
                pinned_indexes.add(index)
                break

    selected = set(sorted(pinned_indexes)[-limit:])
    for index in range(len(rows) - 1, -1, -1):
        if len(selected) >= limit:
            break
        if index not in selected:
            selected.add(index)
    return [rows[index] for index in sorted(selected)]


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
        _ensure_schema()

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
        fact_frames: dict[str, dict[str, Any]] | None = None,
        fact_requirements: list[dict[str, Any]] | None = None,
        execution_plan: list[str] | None = None,
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

        constraint_source = _structured_plan_global_constraint_text(objective) if execution_plan else objective
        current_constraints = _extract_constraints(constraint_source, policy_note)
        constraints: list[str] = []
        for item in [*carried_constraints, *current_constraints]:
            if item and item not in constraints:
                constraints.append(item)

        current_requirements = _clean_requirements(requirements or [], self.limits["requirement_store_items"])
        if continuation:
            previous_requirements = _clean_requirements(list(previous.get("requirements") or []), self.limits["requirement_store_items"])
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
            state_requirements = merged_requirements[-self.limits["requirement_store_items"]:]
        else:
            state_requirements = current_requirements

        try:
            from .goals import get_goal_record
            persistent_goal = get_goal_record(self._cid())
        except Exception:
            persistent_goal = {}

        state.update({
            "turn_id": max(0, int(turn_id)),
            "status": "active",
            "objective": _clip(objective, self.limits["objective_chars"]),
            "persistent_goal": {
                "goal": _clip(persistent_goal.get("goal"), 1200),
                "definition_of_done": _clip(persistent_goal.get("definition_of_done"), 900),
                "status": _clip(persistent_goal.get("status"), 24),
            } if persistent_goal else {},
            "task_frame": dict(task_frame or {}),
            "fact_frames": {str(key): dict(value or {}) for key, value in dict(fact_frames or {}).items() if isinstance(value, dict)},
            "fact_requirements": [dict(item) for item in list(fact_requirements or []) if isinstance(item, dict)][:16],
            "background": {
                "rolling_summary": _clip(rolling_summary, self.limits["background_chars"]) if continuation else "",
                # Avoid stale task leakage. Raw recent conversational setup is
                # only carried when the current request explicitly refers back.
                "recent_context": _recent_context(recent_messages, self.limits["recent_context_chars"]) if continuation else "",
                "recalled_context": _clip(recalled_context, self.limits["memory_chars"]),
            },
            "constraints": constraints[:32],
            "requirements": state_requirements,
            "tool_capabilities": _tool_capabilities(tool_schemas),
        })
        if execution_plan:
            self._install_scheduler(state, objective, execution_plan)
        _save(state, self._cid())
        return state

    def _install_scheduler(self, state: dict[str, Any], objective: str, steps: list[str]) -> None:
        """Install a harness-owned sequential plan without exposing future steps.

        The full plan is durable state used by the deterministic scheduler.  The
        model-facing :meth:`render` projection deliberately includes only the
        active step, so later requirements cannot contaminate tool selection or
        tempt the main model to jump ahead.
        """
        clean_steps: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in list(steps or [])[: self.limits["scheduler_steps"]]:
            task = _clip(_normalize_space(raw), self.limits["scheduler_task_chars"])
            marker = task.casefold()
            if not task or marker in seen:
                continue
            seen.add(marker)
            clean_steps.append({
                "id": f"step-{len(clean_steps) + 1:03d}",
                "task": task,
                "status": "PENDING",
                "reason": "",
                "result": "",
            })
        state["scheduler"] = {
            "enabled": len(clean_steps) >= 2,
            "overall_objective": _clip(objective, self.limits["scheduler_objective_chars"]),
            "active_index": 0,
            "steps": clean_steps,
        }

    def set_execution_plan(self, objective: str, steps: list[str]) -> dict[str, Any]:
        state = _load(self._cid())
        self._install_scheduler(state, objective, steps)
        _save(state, self._cid())
        return copy.deepcopy(state.get("scheduler") or {})

    def active_requirement(self, *, state: dict[str, Any] | None = None) -> str:
        state = _load(self._cid()) if state is None else state
        scheduler = dict(state.get("scheduler") or {})
        if not scheduler.get("enabled"):
            return ""
        steps = list(scheduler.get("steps") or [])
        index = int(scheduler.get("active_index") or 0)
        if 0 <= index < len(steps) and isinstance(steps[index], dict):
            return str(steps[index].get("task") or "")
        return ""

    def scheduler_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(dict(_load(self._cid()).get("scheduler") or {}))

    def mark_active_requirement(
        self, status: str, *, reason: str = "", result: str = "",
    ) -> dict[str, Any]:
        """Close the active scheduled requirement and advance exactly one step.

        Only terminal ``PASS``/``FAIL`` states are accepted.  Advancing is a
        harness operation; model text cannot move the pointer directly.
        """
        normalized = str(status or "").upper()
        if normalized not in {"PASS", "FAIL"}:
            raise ValueError("scheduled requirement status must be PASS or FAIL")
        state = _load(self._cid())
        scheduler = dict(state.get("scheduler") or {})
        if not scheduler.get("enabled"):
            return scheduler
        steps = [dict(item) for item in list(scheduler.get("steps") or []) if isinstance(item, dict)]
        index = int(scheduler.get("active_index") or 0)
        if not (0 <= index < len(steps)):
            return scheduler
        steps[index]["status"] = normalized
        steps[index]["reason"] = _clip(reason, 180)
        steps[index]["result"] = _clip(result, self.limits["scheduler_result_chars"])
        if index + 1 < len(steps):
            scheduler["active_index"] = index + 1
        else:
            scheduler["active_index"] = len(steps)
        scheduler["steps"] = steps
        state["scheduler"] = scheduler
        _save(state, self._cid())
        return copy.deepcopy(scheduler)

    def scheduler_complete(self, *, state: dict[str, Any] | None = None) -> bool:
        state = _load(self._cid()) if state is None else state
        scheduler = dict(state.get("scheduler") or {})
        if not scheduler.get("enabled"):
            return True
        steps = list(scheduler.get("steps") or [])
        return bool(steps) and all(
            isinstance(item, dict) and str(item.get("status") or "") in {"PASS", "FAIL"}
            for item in steps
        )

    def render_scheduler_results(self, max_chars: int = 6000) -> str:
        """Render completed scheduler results for the final synthesis pass.

        This is intentionally separate from the normal canonical state so future
        steps remain invisible during execution. It is used only after all steps
        are terminal. Prefer compacting every row over dropping early rows: the
        final synthesis must be able to account for the complete plan.
        """
        state = _load(self._cid())
        scheduler = dict(state.get("scheduler") or {})
        if not scheduler.get("enabled") or not self.scheduler_complete(state=state):
            return "[]"
        limit = max(800, int(max_chars))
        source_rows = [dict(item) for item in list(scheduler.get("steps") or []) if isinstance(item, dict)]

        def encode(task_chars: int, reason_chars: int, result_chars: int) -> str:
            rows = [{
                "id": str(item.get("id") or ""),
                "task": _clip(item.get("task"), task_chars),
                "status": str(item.get("status") or ""),
                "reason": _clip(item.get("reason"), reason_chars),
                "result": _clip(item.get("result"), result_chars),
            } for item in source_rows]
            return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))

        # Progressive whole-plan compaction. Even the smallest tier retains each
        # step's identity/status plus enough task/result text to synthesize a
        # coverage table or blocker summary.
        for task_chars, reason_chars, result_chars in (
            (360, 160, 520),
            (220, 120, 300),
            (150, 90, 180),
            (100, 70, 110),
            (72, 48, 72),
        ):
            text = encode(task_chars, reason_chars, result_chars)
            if len(text) <= limit:
                return text

        # Extremely small caller budgets cannot represent a large plan in full.
        # Preserve deterministic coverage metadata for every step before falling
        # back to dropping rows.
        minimal_rows = [{
            "id": str(item.get("id") or ""),
            "status": str(item.get("status") or ""),
            "task": _clip(item.get("task"), 48),
        } for item in source_rows]
        text = json.dumps(minimal_rows, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= limit:
            return text
        while len(minimal_rows) > 1 and len(text) > limit:
            minimal_rows.pop(0)
            text = json.dumps(minimal_rows, ensure_ascii=False, separators=(",", ":"))
        return text if len(text) <= limit else "[]"

    def update_tools(self, tool_schemas: list[dict[str, Any]]) -> None:
        state = _load(self._cid())
        state["tool_capabilities"] = _tool_capabilities(tool_schemas)
        _save(state, self._cid())

    def update_requirements(self, requirements: list[dict[str, Any]]) -> None:
        state = _load(self._cid())
        state["requirements"] = _clean_requirements(requirements, self.limits["requirement_store_items"])
        _save(state, self._cid())

    def update_fact_requirements(self, requirements: list[dict[str, Any]]) -> None:
        state = _load(self._cid())
        state["fact_requirements"] = [dict(item) for item in list(requirements or []) if isinstance(item, dict)][:16]
        _save(state, self._cid())

    def update_active_context(
        self,
        *,
        task_frame: dict[str, Any] | None = None,
        fact_frames: dict[str, dict[str, Any]] | None = None,
        fact_requirements: list[dict[str, Any]] | None = None,
    ) -> None:
        """Atomically replace model-facing execution context for the active step.

        Scheduler advancement and this projection update are harness-owned.  This
        prevents a newly active requirement from being paired with stale frames
        from the previous step in the next main-model prompt.
        """
        state = _load(self._cid())
        state["task_frame"] = dict(task_frame or {})
        state["fact_frames"] = {
            str(key): dict(value or {})
            for key, value in dict(fact_frames or {}).items()
            if isinstance(value, dict)
        }
        state["fact_requirements"] = [
            dict(item) for item in list(fact_requirements or []) if isinstance(item, dict)
        ][:16]
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
        grounding = grounding_metadata(
            tool_name, result_text, arguments=arguments,
            task_frame=dict(state.get("task_frame") or {}),
            fact_frames=dict(state.get("fact_frames") or {}),
        )
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
            # Scope derived from successful numeric quote rows in the original
            # (unclipped) result.  The human-readable evidence preview may be
            # too short to contain every instrument in a multi-quote response.
            "market_instruments": grounding.get("market_instruments", []),
            # Deterministic validator-only proof derived from the full result
            # before evidence previews are clipped. This is intentionally omitted
            # from model-facing render() output to avoid prompt bloat.
            "grounding_proof": grounding.get("grounding_proof", {}),
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
            state["verified_observations"] = _bounded_observations(
                state, observations, self.limits["evidence_items"]
            )
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
        # A structured scheduler is the authoritative completion contract for a
        # compiled multi-step turn.  Deterministic fast paths and generic final
        # answer helpers may close an atomic subtask, but they must never mark
        # the whole turn complete while scheduled requirements remain pending.
        # ``blocked=True`` is still allowed to terminate the turn explicitly for
        # hard runtime/safety failures.
        if not blocked and not self.scheduler_complete(state=state):
            state["status"] = "active"
            state["current_plan"] = []
            # The unfinished scheduler state is compact and durable; native tool
            # schemas are still turn-scoped and must be re-routed next turn.
            state["tool_capabilities"] = []
            _save(state, self._cid())
            return
        state["status"] = "blocked" if blocked else "complete"
        state["current_plan"] = []
        # Tool schemas are Tier-1 state.  Keep them while a structured scheduler
        # is still active, but strip them as soon as the turn reaches a terminal
        # state so diagnostics/prompt projections cannot accidentally replay them.
        state["tool_capabilities"] = []
        _save(state, self._cid())

    def render_evidence(
        self, max_chars: int | None = None, *, state: dict[str, Any] | None = None,
        tool_names: set[str] | None = None, fact_types: set[str] | None = None,
        include_recent: int = 0,
    ) -> str:
        """Render bounded untrusted evidence excerpts for a user-role prompt block.

        Callers rebuilding one prompt may pass a snapshot so canonical state and
        evidence are rendered from the same SQLite read.  Structured-plan callers
        should also pass the active step's tool/fact scope.  That keeps durable
        evidence available without replaying unrelated observations from earlier
        scheduler steps on every model request.  ``include_recent`` is a small
        escape hatch for explicit cross-check/reuse steps that intentionally need
        a little prior context.
        """
        state = _load(self._cid()) if state is None else state
        limit = max(400, int(max_chars or self.limits["evidence_render_chars"]))
        wanted_tools = {str(name) for name in (tool_names or set()) if str(name)}
        wanted_facts = {str(name) for name in (fact_types or set()) if str(name)}
        observations = [
            item for item in list(state.get("verified_observations") or [])[-self.limits["evidence_items"]:]
            if isinstance(item, dict)
        ]
        if wanted_tools or wanted_facts:
            selected: list[dict[str, Any]] = []
            selected_ids: set[int] = set()
            for item in observations:
                item_facts = {str(value) for value in (item.get("fact_types") or []) if str(value)}
                if str(item.get("tool") or "") in wanted_tools or bool(item_facts & wanted_facts):
                    selected.append(item)
                    selected_ids.add(id(item))
            if include_recent > 0:
                for item in observations[-max(0, int(include_recent)):]:
                    if id(item) not in selected_ids:
                        selected.append(item)
                        selected_ids.add(id(item))
            observations = selected
        elif include_recent > 0:
            observations = observations[-max(0, int(include_recent)):]

        rows: list[dict[str, Any]] = []
        for item in observations:
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

    def render(
        self, *, include_tool_capabilities: bool = True, state: dict[str, Any] | None = None,
    ) -> str:
        """Render valid bounded canonical metadata JSON.

        The main model already receives native tool schemas, so callers may omit
        the duplicate capability descriptions. The fast validator can retain them.
        A supplied snapshot avoids a second database read when a prompt also needs
        the evidence block.
        """
        state = _load(self._cid()) if state is None else state
        scheduler = dict(state.get("scheduler") or {})
        scheduler_enabled = bool(scheduler.get("enabled"))
        active_index = int(scheduler.get("active_index") or 0)
        scheduler_steps = list(scheduler.get("steps") or [])
        active_step = (
            dict(scheduler_steps[active_index])
            if scheduler_enabled and 0 <= active_index < len(scheduler_steps) and isinstance(scheduler_steps[active_index], dict)
            else {}
        )
        completed_steps = sum(
            1 for item in scheduler_steps
            if isinstance(item, dict) and str(item.get("status") or "") in {"PASS", "FAIL"}
        )
        compact = {
            "turn_id": state.get("turn_id", 0),
            "task_epoch": state.get("task_epoch", 0),
            "task_frame": dict(state.get("task_frame", {}) or {}),
            "fact_frames": {str(key): dict(value or {}) for key, value in dict(state.get("fact_frames", {}) or {}).items() if isinstance(value, dict)},
            "fact_requirements": [dict(item) for item in list(state.get("fact_requirements", []) or []) if isinstance(item, dict)],
            "status": state.get("status", "idle"),
            # The complete user objective remains persisted in scheduler state,
            # but future task text is intentionally withheld from the main-model
            # projection while a compiled plan is active.
            "objective": active_step.get("task", "") if scheduler_enabled else state.get("objective", ""),
            "active_requirement": active_step if scheduler_enabled else {},
            "scheduler": ({
                "enabled": True,
                "active_index": active_index,
                "step_count": len(scheduler_steps),
                "completed_count": completed_steps,
            } if scheduler_enabled else {"enabled": False}),
            "persistent_goal": dict(state.get("persistent_goal", {}) or {}),
            "background": dict(state.get("background", {}) or {}),
            "constraints": list(state.get("constraints", []) or []),
            # Requirement-level evidence excerpts are persisted in SQLite for
            # auditability, but are intentionally omitted from the model-facing
            # canonical state. The model already receives a separately bounded
            # observation evidence block.
            # Full requirement rows remain durable for audit/finalization, but
            # are hidden during scheduled execution because they may contain
            # future task strings/tool names. The active requirement and native
            # schema set are the only immediate execution contract.
            "requirements": [] if scheduler_enabled else _clean_requirements(
                list(state.get("requirements", []) or []), self.limits["requirement_items"], evidence_preview_chars=0,
            ),
            "tool_capabilities": list(state.get("tool_capabilities", []) or []) if include_tool_capabilities else [],
            # In scheduler mode the observation excerpts are already rendered in
            # the separately bounded evidence block.  Duplicating every prior
            # observation here made the mutable system prefix grow across steps,
            # increasing local-model prefill until it crossed the transport
            # timeout.  Ordinary single-task turns retain the historical view.
            "verified_observations": [] if scheduler_enabled else [
                {key: value for key, value in item.items() if key not in {"evidence_preview", "grounding_proof"}}
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
                "persistent_goal": compact.get("persistent_goal", {}),
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
        _invalidate_state_cache(self._cid())
