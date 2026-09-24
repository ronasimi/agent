"""Timestamp-aware cross-conversation recall for the context hierarchy.

Historical recall is intentionally deterministic: relative date phrases are
resolved by the harness in the configured timezone, then SQLite/FTS5 retrieves a
small evidence set. The 4B model only summarizes already-selected history.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .config import load_config
from .memory import search_conversation_history_records

_AGENT_CFG = load_config().get("agent", {})
_DEFAULT_TIMEZONE = str(_AGENT_CFG.get("timezone") or "UTC")
_RECALL_LIMIT = max(4, min(int((_AGENT_CFG.get("context") or {}).get("historical_recall_messages", 32)), 80))
_RECALL_CHARS = max(1200, min(int((_AGENT_CFG.get("context") or {}).get("historical_recall_chars", 7000)), 16000))

_RECALL_RE = re.compile(
    r"\b(?:"
    r"what\s+did\s+(?:we|i|you)\s+(?:talk|discuss|cover|work|ask|say|tell)|"
    r"what\s+were\s+we\s+(?:talking|discussing|working)\s+about|"
    r"what\s+have\s+we\s+(?:talked|discussed|covered)|"
    r"do\s+you\s+remember\s+what\s+(?:we|i|you)\s+(?:talked|discussed|said|asked|covered)|"
    r"remember\s+what\s+(?:we|i|you)\s+(?:talked|discussed|said|asked|covered)|"
    r"remember\s+(?:our|the)\s+(?:chat|conversation|discussion)|"
    r"recall\s+(?:our|the)\s+(?:chat|conversation|discussion)|"
    r"(?:search|find|look\s+through|check)\s+(?:our|my|the)?\s*(?:chat|conversation)(?:\s+history)?|"
    r"(?:our|the|a)\s+(?:previous|earlier|old)\s+(?:chat|conversation|discussion)|"
    r"(?:yesterday|last\s+(?:night|week)|earlier\s+this\s+week)\b.{0,80}\b(?:talk|discuss|chat|conversation|asked|said|told)"
    r")\b",
    re.I,
)

_TEMPORAL_NOISE_RE = re.compile(
    r"\b(?:today|yesterday|last\s+night|last\s+week|earlier\s+this\s+week|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)
_RECALL_NOISE_RE = re.compile(
    r"\b(?:what|did|were|was|we|i|you|our|my|the|a|an|about|talk|talked|talking|"
    r"discuss|discussed|discussing|conversation|chat|history|remember|recall|search|"
    r"find|look|through|check|asked|ask|said|say|told|tell|covered|cover|worked|working)\b",
    re.I,
)
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_ACTION_TAIL_RE = re.compile(
    r"(?:[,;.]|\band\b|\bthen\b|\balso\b|\bafter\s+that\b)\s*"
    r"(?P<tail>(?:check|inspect|open|edit|write|run|execute|browse|visit|download|"
    r"create|delete|send|schedule|remind|compare|test|fix|update)\b.*)$",
    re.I,
)


def is_historical_recall_request(user_text: str) -> bool:
    return bool(_RECALL_RE.search(" ".join(str(user_text or "").split())))


def historical_recall_tool_text(user_text: str) -> str:
    """Return only an explicit post-recall action for tool selection.

    Topic words inside the recall clause are evidence-search terms, not current
    environment actions. This prevents a 4B model from receiving file/system
    tools just because an old conversation happened to be about Ollama or a
    repository. Compound requests retain tools for an explicit action tail.
    """
    text = " ".join(str(user_text or "").split())
    if not is_historical_recall_request(text):
        return text
    match = _ACTION_TAIL_RE.search(text)
    return str(match.group("tail") or "").strip() if match else ""


def _local_midnight(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _to_db_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _iso_local(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def resolve_historical_window(
    user_text: str, *, timezone_name: str = "", now: datetime | None = None,
) -> dict[str, str]:
    """Resolve common relative-date phrases to an exact half-open local window."""
    tz_name = str(timezone_name or _DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
        tz_name = "UTC"
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    else:
        current = current.astimezone(tz)
    lower = str(user_text or "").lower()
    start: datetime | None = None
    end: datetime | None = None
    label = "all saved history"

    exact = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", lower)
    if exact:
        try:
            start = datetime(int(exact.group(1)), int(exact.group(2)), int(exact.group(3)), tzinfo=tz)
            end = start + timedelta(days=1)
            label = exact.group(0)
        except ValueError:
            start = end = None
    elif "yesterday" in lower:
        end = _local_midnight(current)
        start = end - timedelta(days=1)
        label = "yesterday"
    elif "last night" in lower:
        today = _local_midnight(current)
        start = today - timedelta(hours=6)  # previous day 18:00
        end = today + timedelta(hours=6)
        label = "last night"
    elif "last week" in lower:
        this_monday = _local_midnight(current) - timedelta(days=current.weekday())
        end = this_monday
        start = end - timedelta(days=7)
        label = "last week"
    elif "earlier this week" in lower:
        start = _local_midnight(current) - timedelta(days=current.weekday())
        end = current
        label = "earlier this week"
    elif re.search(r"\btoday\b", lower):
        start = _local_midnight(current)
        end = start + timedelta(days=1)
        label = "today"
    else:
        for name, weekday in _WEEKDAYS.items():
            if re.search(rf"\b{name}\b", lower):
                days_back = (current.weekday() - weekday) % 7
                if days_back == 0:
                    days_back = 7
                start = _local_midnight(current) - timedelta(days=days_back)
                end = start + timedelta(days=1)
                label = name
                break

    result = {"timezone": tz_name, "label": label, "start_utc": "", "end_utc": "", "start_local": "", "end_local": ""}
    if start is not None:
        result["start_utc"] = _to_db_utc(start)
        result["start_local"] = _iso_local(start)
    if end is not None:
        result["end_utc"] = _to_db_utc(end)
        result["end_local"] = _iso_local(end)
    return result


def _topic_query(user_text: str) -> str:
    text = _TEMPORAL_NOISE_RE.sub(" ", str(user_text or ""))
    text = _RECALL_NOISE_RE.sub(" ", text)
    return " ".join(token for token in re.findall(r"[A-Za-z0-9_.:-]+", text) if len(token) >= 3)[:500]


def _localize_utc_timestamp(value: str, timezone_name: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return parsed.astimezone(ZoneInfo(timezone_name)).isoformat(timespec="seconds")
    except Exception:
        return value


def historical_recall_payload(
    user_text: str, *, timezone_name: str = "", now: datetime | None = None,
    limit: int | None = None, before_id: int = 0,
) -> dict[str, Any] | None:
    if not is_historical_recall_request(user_text):
        return None
    window = resolve_historical_window(user_text, timezone_name=timezone_name, now=now)
    query = _topic_query(user_text)
    # Date-only recap requests need a wider candidate pool than topical FTS
    # searches so we can choose representative excerpts across the requested
    # period without loading the full transcript into the model.
    requested_limit = int(limit or _RECALL_LIMIT)
    search_limit = min(100, max(requested_limit, requested_limit * 3 if not query else requested_limit))
    rows = search_conversation_history_records(
        query,
        start_time=window.get("start_utc", ""),
        end_time=window.get("end_utc", ""),
        limit=search_limit,
        before_id=before_id,
    )
    if len(rows) > requested_limit:
        # Evenly sample the already-bounded chronological evidence so a recap
        # represents the whole period rather than only its last few exchanges.
        if requested_limit <= 1:
            rows = rows[-1:]
        else:
            span = len(rows) - 1
            indexes = sorted({round(i * span / (requested_limit - 1)) for i in range(requested_limit)})
            rows = [rows[i] for i in indexes]
    tz_name = window.get("timezone") or _DEFAULT_TIMEZONE
    matches = []
    for row in rows:
        content = " ".join(str(row.get("content") or "").split())
        if not content:
            continue
        matches.append({
            "conversation_id": row.get("conversation_id", ""),
            "title": row.get("title", "Conversation"),
            "role": row.get("role", ""),
            "created_at": row.get("created_at", ""),
            "created_at_local": _localize_utc_timestamp(str(row.get("created_at") or ""), tz_name),
            "content": content[:900],
        })
    return {
        "kind": "historical_conversation_recall",
        "query": query,
        "resolved_window": window,
        "matches": matches,
    }


def _fit_payload(payload: dict[str, Any], max_chars: int) -> str:
    """Fit recall evidence while always returning valid JSON."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text

    rows = [dict(row) for row in list(payload.get("matches") or [])]
    # First shrink excerpts; preserving more timestamps/turns is usually more
    # useful for recap than keeping a few very long messages verbatim.
    for content_limit in (600, 420, 280, 180):
        for row in rows:
            row["content"] = str(row.get("content") or "")[:content_limit]
        payload["matches"] = rows
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= max_chars:
            return text

    # Then reduce the number of rows while keeping a representative spread.
    while len(rows) > 2:
        target = max(2, len(rows) - 1)
        span = len(rows) - 1
        indexes = sorted({round(i * span / (target - 1)) for i in range(target)})
        rows = [rows[i] for i in indexes]
        payload["matches"] = rows
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= max_chars:
            return text
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_historical_recall_context(
    user_text: str, *, timezone_name: str = "", now: datetime | None = None, before_id: int = 0,
) -> str:
    """Return a bounded timestamped history block only for explicit recall intents."""
    payload = historical_recall_payload(
        user_text, timezone_name=timezone_name, now=now, before_id=before_id,
    )
    if payload is None:
        return ""
    return _fit_payload(payload, _RECALL_CHARS)
