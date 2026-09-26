"""Persistent routing calibration and compact candidate-scoring helpers.

The actual route choice is made by :mod:`al_agent.deterministic_router`. This module
contains durable outcome statistics plus the cheap lexical scoring helpers used to
keep the resident main model's active schema set bounded.
"""

from __future__ import annotations

import math
from functools import lru_cache
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import threading
from dataclasses import dataclass
from typing import Any, Iterable

from tools.runtime import DB_PATH, DB_TIMEOUT, utc_now

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "get", "give", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "please", "show", "that", "the", "then", "this", "to", "use",
    "using", "want", "what", "when", "where", "which", "with", "you",
    # Generic task-control verbs/formatting words add noise to catalog routing.
    "actual", "again", "clearly", "current", "include", "one", "read",
    "report", "result", "results", "retrieve", "run", "same", "state",
    "step", "tool", "usage",
}

# Small deterministic lexical normalization table used only for catalog routing.
# This is intentionally not an embedding/model call: it covers ordinary user
# vocabulary that differs from provider/tool naming (for example "email" vs
# "gmail messages") while keeping routing latency effectively unchanged.
_TOKEN_CANONICAL = {
    "emails": "email",
    "messages": "message",
    "calendars": "calendar",
    "events": "event",
    "files": "file",
    "screenshots": "screenshot",
}
_TOKEN_ALIASES = {
    "email": {"gmail", "mail", "message"},
    "inbox": {"gmail", "mail", "message", "email"},
    "mailbox": {"gmail", "mail", "message", "email"},
    "gmail": {"email", "mail", "message", "inbox"},
    "calendar": {"event", "schedule"},
    "schedule": {"calendar", "event"},
    "drive": {"file", "google"},
    "screenshot": {"capture", "page"},
    "profile": {"identity", "personal", "preference", "name"},
    "identity": {"profile", "personal", "name"},
    "name": {"profile", "identity"},
    "preferences": {"profile", "preference"},
    "preference": {"profile", "preferences"},
}

# Routing remains primarily request/schema based. Learned evidence may move a
# candidate by at most +/- 0.15.
LEARNED_MAX_ADJUSTMENT = 0.15
GLOBAL_ALPHA = 0.10
CONTEXT_ALPHA = 0.18
HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.65
MEDIUM_MAX_TOOLS = 3
LEARNING_HALF_LIFE_DAYS = 30.0

_DB_LOCK = threading.RLock()


@dataclass(frozen=True)
class RankedTool:
    name: str
    score: float
    base_score: float
    description: str
    context_key: str


@dataclass(frozen=True)
class RoutingDecision:
    selected: tuple[str, ...]
    confidence: float
    tier: str
    candidates: tuple[RankedTool, ...]
    context_key: str


@lru_cache(maxsize=1024)
def _tokens(text: str) -> tuple[str, ...]:
    raw = [
        _TOKEN_CANONICAL.get(t, t)
        for t in _TOKEN_RE.findall(str(text or "").lower())
        if t not in _STOP
    ]
    expanded: list[str] = []
    seen: set[str] = set()
    for token in raw:
        for candidate in (token, *sorted(_TOKEN_ALIASES.get(token, set()))):
            canonical = _TOKEN_CANONICAL.get(candidate, candidate)
            if canonical in _STOP or canonical in seen:
                continue
            seen.add(canonical)
            expanded.append(canonical)
    return tuple(expanded)


def _context_key(text: str) -> str:
    # A stable, intentionally coarse lexical context lets similar requests share
    # evidence without storing user prose in the routing database.
    terms = sorted(set(_tokens(text)))[:8]
    return " ".join(terms) or "_general"


def _decayed_ema(value: float, updated_at: str | None) -> float:
    """Decay learned evidence toward neutral (0.5) as it ages."""
    try:
        stamp = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0)
    except Exception:
        return float(value)
    retention = math.pow(0.5, age_days / LEARNING_HALF_LIFE_DAYS)
    return 0.5 + (float(value) - 0.5) * retention


def _schema_parts(schema: dict) -> tuple[str, str, dict]:
    fn = schema.get("function") if isinstance(schema, dict) else {}
    if not isinstance(fn, dict):
        fn = {}
    return (
        str(fn.get("name") or ""),
        str(fn.get("description") or ""),
        fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {},
    )


def _base_score(text: str, schema: dict) -> float:
    name, description, parameters = _schema_parts(schema)
    if not name:
        return 0.0
    q_tokens = set(_tokens(text))
    name_tokens = set(_tokens(name.replace("_", " ")))
    desc_tokens = set(_tokens(description))
    if not q_tokens:
        return 0.0
    # Argument compatibility is only a tie-breaker.  An unrelated schema must
    # never receive a non-zero routing score merely because it has no required
    # arguments; otherwise casual conversation activates arbitrary tools.
    if not (q_tokens & (name_tokens | desc_tokens)):
        return 0.0

    name_coverage = len(q_tokens & name_tokens) / max(1, len(name_tokens))
    query_coverage = len(q_tokens & (name_tokens | desc_tokens)) / max(1, len(q_tokens))
    # Exact natural-language tool-name phrase is highly diagnostic while still
    # working for newly registered tools without code changes.
    normalized_text = " ".join(_tokens(text))
    normalized_name = " ".join(_tokens(name.replace("_", " ")))
    phrase_bonus = 0.15 if normalized_name and normalized_name in normalized_text else 0.0

    required = parameters.get("required") if isinstance(parameters, dict) else []
    required = required if isinstance(required, list) else []
    props = parameters.get("properties") if isinstance(parameters, dict) else {}
    props = props if isinstance(props, dict) else {}
    if not required:
        argument_compatibility = 1.0
    else:
        mentioned = 0
        for field in required:
            field_tokens = set(_tokens(str(field).replace("_", " ")))
            if field_tokens and field_tokens & q_tokens:
                mentioned += 1
        # Do not heavily punish natural-language requests that provide values
        # without spelling JSON field names.
        argument_compatibility = 0.45 + 0.55 * (mentioned / max(1, len(required)))

    # Description specificity: matching a rare-ish schema token should help,
    # but name coverage remains the strongest signal.
    desc_overlap = len(q_tokens & desc_tokens) / max(1, min(len(q_tokens), 8))
    score = (
        0.52 * name_coverage
        + 0.22 * query_coverage
        + 0.11 * min(1.0, desc_overlap)
        + 0.10 * argument_compatibility
        + phrase_bonus
    )
    return max(0.0, min(1.0, score))


class RoutingFeedbackStore:
    """Durable online routing statistics stored in the main harness database."""

    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or DB_PATH)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=DB_TIMEOUT)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with _DB_LOCK, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tool_routing_stats (
                    tool_name TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    ema_success REAL NOT NULL DEFAULT 0.5,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_context_routing_stats (
                    tool_name TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    ema_success REAL NOT NULL DEFAULT 0.5,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tool_name, context_key)
                );
                CREATE TABLE IF NOT EXISTS tool_routing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    outcome REAL,
                    event_type TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_tool_routing_events_tool_time
                    ON tool_routing_events(tool_name, created_at DESC);
                """
            )

    def score_map(self, tool_names: Iterable[str], context_key: str) -> dict[str, tuple[float, float]]:
        names = [str(name) for name in tool_names if str(name)]
        if not names:
            return {}
        placeholders = ",".join("?" for _ in names)
        with _DB_LOCK, self._connect() as conn:
            global_rows = conn.execute(
                f"SELECT tool_name,ema_success,updated_at FROM tool_routing_stats WHERE tool_name IN ({placeholders})",
                names,
            ).fetchall()
            context_rows = conn.execute(
                f"SELECT tool_name,ema_success,updated_at FROM tool_context_routing_stats WHERE context_key=? AND tool_name IN ({placeholders})",
                [context_key, *names],
            ).fetchall()
        global_map = {str(row[0]): _decayed_ema(float(row[1]), row[2]) for row in global_rows}
        context_map = {str(row[0]): _decayed_ema(float(row[1]), row[2]) for row in context_rows}
        return {
            name: (global_map.get(name, 0.5), context_map.get(name, 0.5))
            for name in names
        }

    def scores(self, tool_name: str, context_key: str) -> tuple[float, float]:
        return self.score_map([tool_name], context_key).get(tool_name, (0.5, 0.5))

    def record(
        self,
        tool_name: str,
        context_key: str,
        outcome: float | None,
        *,
        event_type: str,
        detail: str = "",
        global_alpha: float = GLOBAL_ALPHA,
        context_alpha: float = CONTEXT_ALPHA,
    ) -> None:
        """Persist one routing observation immediately.

        outcome=None records diagnostics (e.g. infrastructure timeout) without
        changing confidence.  This prevents transport failures from teaching the
        router that the selected capability itself was wrong.
        """
        now = utc_now()
        clean_tool = str(tool_name or "").strip()
        clean_context = str(context_key or "_general")[:240]
        if not clean_tool:
            return
        numeric = None if outcome is None else max(0.0, min(1.0, float(outcome)))
        with _DB_LOCK, self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_routing_events(created_at,tool_name,context_key,outcome,event_type,detail) VALUES(?,?,?,?,?,?)",
                (now, clean_tool, clean_context, numeric, str(event_type), str(detail)[:1000]),
            )
            if numeric is None:
                return
            row = conn.execute(
                "SELECT attempts,successes,ema_success FROM tool_routing_stats WHERE tool_name=?",
                (clean_tool,),
            ).fetchone()
            attempts = int(row[0]) if row else 0
            successes = int(row[1]) if row else 0
            ema = float(row[2]) if row else 0.5
            ema = global_alpha * numeric + (1.0 - global_alpha) * ema
            conn.execute(
                "INSERT INTO tool_routing_stats(tool_name,attempts,successes,ema_success,updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(tool_name) DO UPDATE SET attempts=excluded.attempts,successes=excluded.successes,ema_success=excluded.ema_success,updated_at=excluded.updated_at",
                (clean_tool, attempts + 1, successes + int(numeric >= 0.5), ema, now),
            )
            row = conn.execute(
                "SELECT attempts,successes,ema_success FROM tool_context_routing_stats WHERE tool_name=? AND context_key=?",
                (clean_tool, clean_context),
            ).fetchone()
            attempts = int(row[0]) if row else 0
            successes = int(row[1]) if row else 0
            ema = float(row[2]) if row else 0.5
            ema = context_alpha * numeric + (1.0 - context_alpha) * ema
            conn.execute(
                "INSERT INTO tool_context_routing_stats(tool_name,context_key,attempts,successes,ema_success,updated_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(tool_name,context_key) DO UPDATE SET attempts=excluded.attempts,successes=excluded.successes,ema_success=excluded.ema_success,updated_at=excluded.updated_at",
                (clean_tool, clean_context, attempts + 1, successes + int(numeric >= 0.5), ema, now),
            )
