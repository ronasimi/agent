"""Durable browser UI state and trajectory storage.

The live Playwright objects stay in :mod:`tools.browser_ui`; this module stores
only bounded, serializable state needed for recovery, diagnostics, and
benchmarking.  It intentionally uses the agent's existing SQLite database so
browser trajectories survive process restarts without leaking into chat
history/context.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .conversation_context import get_active_conversation_id, normalize_conversation_id

_SCHEMA_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_path() -> str:
    return os.environ.get("AGENT_DB_PATH", "/app/memory/knowledge.db")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=float(os.environ.get("AGENT_DB_TIMEOUT", "15")))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _load_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _bounded(value: Any, limit: int = 12000) -> Any:
    """Bound audit payloads without discarding their high-value structure."""
    try:
        encoded = _json(value)
    except (TypeError, ValueError):
        encoded = str(value)
    if len(encoded) <= limit:
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in (
            "ok", "operation", "state_version", "changed", "error", "verification",
            "scores", "metrics", "delta", "safety", "recovery",
        ):
            if key in value:
                result[key] = value[key]
        result["_truncated"] = True
        return result
    return str(value)[:limit]


class BrowserStateStore:
    """Persist canonical browser state, UI requirements, and action trajectories."""

    def __init__(self, conversation_id: str | None = None) -> None:
        self.conversation_id = normalize_conversation_id(conversation_id or get_active_conversation_id())
        self._ensure_schema()

    @staticmethod
    def _ensure_schema() -> None:
        with _SCHEMA_LOCK, _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS browser_ui_state (
                    conversation_id TEXT PRIMARY KEY,
                    state_version INTEGER NOT NULL DEFAULT 0,
                    state_json TEXT NOT NULL DEFAULT '{}',
                    requirements_json TEXT NOT NULL DEFAULT '[]',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS browser_ui_trajectory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    state_version INTEGER NOT NULL DEFAULT 0,
                    operation TEXT NOT NULL,
                    action_json TEXT NOT NULL DEFAULT '{}',
                    process_success INTEGER NOT NULL DEFAULT 0,
                    outcome_success INTEGER,
                    timings_json TEXT NOT NULL DEFAULT '{}',
                    token_metrics_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_browser_ui_trajectory_conv_step
                  ON browser_ui_trajectory(conversation_id, step_index);
                """
            )

    def load(self) -> dict[str, Any]:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM browser_ui_state WHERE conversation_id = ?",
                (self.conversation_id,),
            ).fetchone()
        if row is None:
            return {
                "conversation_id": self.conversation_id,
                "state_version": 0,
                "state": {},
                "requirements": [],
                "metrics": {},
            }
        return {
            "conversation_id": self.conversation_id,
            "state_version": int(row["state_version"] or 0),
            "state": _load_json(row["state_json"], {}),
            "requirements": _load_json(row["requirements_json"], []),
            "metrics": _load_json(row["metrics_json"], {}),
            "updated_at": row["updated_at"],
        }

    def save_state(
        self,
        *,
        state_version: int,
        state: dict[str, Any],
        requirements: list[dict[str, Any]] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        previous = self.load()
        requirements_value = previous.get("requirements", []) if requirements is None else list(requirements)
        metrics_value = previous.get("metrics", {}) if metrics is None else dict(metrics)
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO browser_ui_state(
                    conversation_id, state_version, state_json, requirements_json, metrics_json, updated_at
                ) VALUES(?,?,?,?,?,?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    state_version=excluded.state_version,
                    state_json=excluded.state_json,
                    requirements_json=excluded.requirements_json,
                    metrics_json=excluded.metrics_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self.conversation_id,
                    max(0, int(state_version)),
                    _json(_bounded(state, 80000)),
                    _json(_bounded(requirements_value, 20000)),
                    _json(_bounded(metrics_value, 12000)),
                    _utc_now(),
                ),
            )

    def set_requirements(self, requirements: list[dict[str, Any]]) -> None:
        current = self.load()
        self.save_state(
            state_version=current.get("state_version", 0),
            state=dict(current.get("state") or {}),
            requirements=list(requirements or []),
            metrics=dict(current.get("metrics") or {}),
        )

    def record_step(
        self,
        *,
        state_version: int,
        operation: str,
        action: dict[str, Any],
        process_success: bool,
        outcome_success: bool | None,
        timings: dict[str, Any],
        token_metrics: dict[str, Any],
        result: dict[str, Any],
    ) -> int:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(step_index), 0) AS n FROM browser_ui_trajectory WHERE conversation_id = ?",
                (self.conversation_id,),
            ).fetchone()
            step_index = int(row["n"] or 0) + 1
            cursor = conn.execute(
                """
                INSERT INTO browser_ui_trajectory(
                    conversation_id, step_index, state_version, operation, action_json,
                    process_success, outcome_success, timings_json, token_metrics_json,
                    result_json, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    self.conversation_id,
                    step_index,
                    max(0, int(state_version)),
                    str(operation or "")[:32],
                    _json(_bounded(action, 6000)),
                    1 if process_success else 0,
                    None if outcome_success is None else (1 if outcome_success else 0),
                    _json(_bounded(timings, 6000)),
                    _json(_bounded(token_metrics, 6000)),
                    _json(_bounded(result, 16000)),
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def trajectory(self, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 2000))
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM browser_ui_trajectory
                WHERE conversation_id = ? ORDER BY step_index DESC LIMIT ?
                """,
                (self.conversation_id, limit),
            ).fetchall()
        result = []
        for row in reversed(rows):
            result.append({
                "id": int(row["id"]),
                "step_index": int(row["step_index"]),
                "state_version": int(row["state_version"]),
                "operation": row["operation"],
                "action": _load_json(row["action_json"], {}),
                "process_success": bool(row["process_success"]),
                "outcome_success": None if row["outcome_success"] is None else bool(row["outcome_success"]),
                "timings": _load_json(row["timings_json"], {}),
                "token_metrics": _load_json(row["token_metrics_json"], {}),
                "result": _load_json(row["result_json"], {}),
                "created_at": row["created_at"],
            })
        return result

    def metrics_summary(self) -> dict[str, Any]:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS steps,
                       SUM(CASE WHEN process_success = 1 THEN 1 ELSE 0 END) AS process_ok,
                       SUM(CASE WHEN outcome_success = 1 THEN 1 ELSE 0 END) AS outcome_ok,
                       SUM(CASE WHEN outcome_success IS NOT NULL THEN 1 ELSE 0 END) AS outcome_checks
                FROM browser_ui_trajectory WHERE conversation_id = ?
                """,
                (self.conversation_id,),
            ).fetchone()
        steps = int(row["steps"] or 0)
        process_ok = int(row["process_ok"] or 0)
        outcome_ok = int(row["outcome_ok"] or 0)
        outcome_checks = int(row["outcome_checks"] or 0)
        return {
            "steps": steps,
            "process_success_rate": (process_ok / steps) if steps else None,
            "outcome_checks": outcome_checks,
            "outcome_success_rate": (outcome_ok / outcome_checks) if outcome_checks else None,
        }
