"""Compact prompt-facing history for the three-tier State Tape lifecycle.

Raw chat/tool rows remain durable in SQLite for audit and search, but they are
never replayed into later model prompts.  Completed turns are reduced to compact
state-tape entries; older resolved entries are folded deterministically into the
conversation rolling summary.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .memory import (
    _connect,
    apply_conversation_compaction,
    ensure_conversation,
    get_conversation_summary,
    set_conversation_summary,
)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "get",
    "grab", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please",
    "the", "this", "to", "use", "with", "you",
}


@dataclass(slots=True, frozen=True)
class StateTapeEntry:
    """One compact durable turn record."""

    id: int
    turn_id: int
    source_message_id: int
    status: str
    objective: str
    summary: str
    unresolved_key: str
    resolved: bool
    created_at: str

    def render(self) -> str:
        prefix = "UNRESOLVED" if not self.resolved else "Turn"
        return f"{prefix} {self.turn_id}: {self.summary}"


@dataclass(slots=True, frozen=True)
class CompactToolOutcome:
    """Protocol-free representation of one active-turn tool result."""

    tool: str
    ok: bool
    summary: str
    observation_id: str = ""


class StateTapeStore:
    """Persist and render Tier-2/Tier-3 prompt state for one conversation."""

    def __init__(
        self,
        conversation_id: str | None = None,
        *,
        recent_entries: int = 6,
        unresolved_entries: int = 3,
        rolling_summary_chars: int = 3200,
        entry_chars: int = 520,
    ) -> None:
        self.conversation_id = ensure_conversation(conversation_id)
        self.recent_entries = max(1, int(recent_entries))
        self.unresolved_entries = max(1, int(unresolved_entries))
        self.rolling_summary_chars = max(800, int(rolling_summary_chars))
        self.entry_chars = max(180, int(entry_chars))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with _connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS state_tape_entries (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       conversation_id TEXT NOT NULL,
                       turn_id INTEGER NOT NULL,
                       source_message_id INTEGER NOT NULL DEFAULT 0,
                       status TEXT NOT NULL DEFAULT 'complete',
                       objective TEXT NOT NULL DEFAULT '',
                       objective_key TEXT NOT NULL DEFAULT '',
                       summary TEXT NOT NULL,
                       unresolved_key TEXT NOT NULL DEFAULT '',
                       resolved INTEGER NOT NULL DEFAULT 1,
                       created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                       updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                       UNIQUE(conversation_id, turn_id)
                   )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_state_tape_conversation ON state_tape_entries(conversation_id, id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_state_tape_unresolved ON state_tape_entries(conversation_id, resolved, id)"
            )

    @staticmethod
    def _single_line(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @classmethod
    def _clip(cls, text: Any, limit: int) -> str:
        value = cls._single_line(text)
        if len(value) <= limit:
            return value
        return value[: max(1, limit - 1)].rstrip() + "…"

    @staticmethod
    def _decode_json_layers(raw: Any) -> Any:
        value = raw
        for _ in range(4):
            if isinstance(value, str):
                stripped = value.strip()
                try:
                    value = json.loads(stripped)
                    continue
                except (TypeError, ValueError, json.JSONDecodeError):
                    return stripped
            if isinstance(value, dict) and "result" in value:
                inner = value.get("result")
                if isinstance(inner, (str, dict, list)):
                    value = inner
                    continue
            break
        return value

    @staticmethod
    def _objective_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9_.:/-]+", str(text or "").casefold())
            if len(token) > 1 and token not in _STOPWORDS
        }

    @classmethod
    def _objective_key(cls, text: str) -> str:
        tokens = sorted(cls._objective_tokens(text))[:24]
        canonical = " ".join(tokens) or cls._single_line(text).casefold()[:240]
        return hashlib.sha1(canonical.encode("utf-8", errors="replace")).hexdigest()[:20]

    @staticmethod
    def _human_bytes(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        unit = units[0]
        for unit in units:
            if abs(number) < 1024 or unit == units[-1]:
                break
            number /= 1024
        return f"{number:.2f}{unit}"

    @staticmethod
    def _weather_label(code: Any) -> str:
        labels = {
            0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "cloudy",
            45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
            61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow",
            73: "snow", 80: "rain showers", 81: "rain showers",
            95: "thunderstorm",
        }
        return labels.get(code, "")

    @classmethod
    def collapse_tool_result(
        cls,
        *,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        result_text: str,
        status: str,
        observation_id: str = "",
    ) -> CompactToolOutcome:
        """Collapse one raw tool response before it can enter historical context."""

        name = str(tool_name or "tool")
        args = dict(arguments or {})
        ok = str(status or "").lower() in {"ok", "success", "partial"}
        payload = cls._decode_json_layers(result_text)
        state = "succeeded" if ok else "failed"

        if name == "weather_forecast" and isinstance(payload, dict):
            current = payload.get("current") if isinstance(payload.get("current"), dict) else {}
            provider = payload.get("provider") or "weather provider"
            when = current.get("time") or payload.get("retrieved_at")
            temp = current.get("temperature_2m")
            label = cls._weather_label(current.get("weather_code"))
            lat = payload.get("latitude", args.get("latitude"))
            lon = payload.get("longitude", args.get("longitude"))
            location = args.get("location") or (
                f"{lat},{lon}" if lat is not None and lon is not None else "requested location"
            )
            details = [f"{temp}°C" if temp is not None else ""]
            if label:
                details.append(label)
            summary = f"Weather {state} for {location} from {provider}"
            if when:
                summary += f" at {when}"
            summary += ": " + ", ".join(item for item in details if item)
            return CompactToolOutcome(name, ok, cls._clip(summary + ".", 300), observation_id)

        if name == "current_time" and isinstance(payload, dict):
            zone = payload.get("timezone") or args.get("timezone_name") or "requested timezone"
            local = payload.get("local") or payload.get("time")
            utc = payload.get("utc")
            detail = ", ".join(
                item for item in (
                    f"local={local}" if local else "",
                    f"UTC={utc}" if utc else "",
                ) if item
            )
            return CompactToolOutcome(
                name, ok, cls._clip(f"Current time {state} for {zone}: {detail or 'time retrieved'}.", 300), observation_id
            )

        if name == "calculate" and isinstance(payload, dict):
            expression = payload.get("expression") or args.get("expression")
            value = payload.get("result")
            return CompactToolOutcome(
                name, ok, cls._clip(f"Calculation {state}: {expression} = {value}.", 260), observation_id
            )

        if name == "memory_info" and isinstance(payload, dict):
            memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else payload
            percent = memory.get("percent")
            used = memory.get("used")
            total = memory.get("total")
            details = []
            if percent is not None:
                details.append(f"{percent}% used")
            if used is not None and total is not None:
                details.append(f"{cls._human_bytes(used)}/{cls._human_bytes(total)}")
            return CompactToolOutcome(name, ok, cls._clip(f"Memory query {state}: {', '.join(details) or 'counters retrieved'}.", 280), observation_id)

        if name == "temperature_sensors" and isinstance(payload, dict):
            preferred: list[str] = []
            for sensor, rows in payload.items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(row.get("current"), (int, float)):
                        continue
                    label = str(row.get("label") or sensor)
                    if label.casefold() in {"cpu", "tctl"} or sensor in {"k10temp", "thinkpad"}:
                        preferred.append(f"{label}={row['current']}°C")
                if len(preferred) >= 3:
                    break
            return CompactToolOutcome(name, ok, cls._clip(f"Temperature query {state}: {', '.join(preferred[:3]) or 'sensor readings retrieved'}.", 280), observation_id)

        if name == "take_web_screenshot":
            target = args.get("url") or "requested page"
            artifact = ""
            if isinstance(payload, dict):
                artifact = str(payload.get("path") or payload.get("file") or payload.get("image_path") or "")
            suffix = f"; artifact={artifact}" if artifact else ""
            return CompactToolOutcome(name, ok, cls._clip(f"Web screenshot {state} for {target}{suffix}.", 300), observation_id)

        if name == "remember":
            fact = args.get("fact") or (payload.get("fact") if isinstance(payload, dict) else "")
            return CompactToolOutcome(name, ok, cls._clip(f"Memory write {state}: {fact or cls._single_line(payload)}.", 280), observation_id)

        # Generic deterministic fallback: retain only small scalar fields or a
        # bounded textual preview.  Never place raw JSON/XML in the tape.
        if isinstance(payload, dict):
            scalar = []
            for key in ("status", "message", "value", "url", "path", "error", "result"):
                value = payload.get(key)
                if value is not None and not isinstance(value, (dict, list)):
                    scalar.append(f"{key}={value}")
            if not scalar:
                for key, value in list(payload.items())[:4]:
                    if not isinstance(value, (dict, list)):
                        scalar.append(f"{key}={value}")
            detail = ", ".join(scalar) or "structured result recorded"
        elif isinstance(payload, list):
            detail = f"{len(payload)} item(s) returned"
        else:
            detail = cls._single_line(payload)
        return CompactToolOutcome(name, ok, cls._clip(f"{name} {state}: {detail}", 300), observation_id)

    @staticmethod
    def _row_to_entry(row: tuple[Any, ...]) -> StateTapeEntry:
        return StateTapeEntry(
            id=int(row[0]), turn_id=int(row[1]), source_message_id=int(row[2] or 0),
            status=str(row[3] or "complete"), objective=str(row[4] or ""),
            summary=str(row[5] or ""), unresolved_key=str(row[6] or ""),
            resolved=bool(row[7]), created_at=str(row[8] or ""),
        )

    def recent(self, limit: int | None = None, *, resolved_only: bool = True) -> list[StateTapeEntry]:
        wanted = max(1, int(limit or self.recent_entries))
        clause = "AND resolved=1" if resolved_only else ""
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT id,turn_id,source_message_id,status,objective,summary,unresolved_key,resolved,created_at "
                f"FROM state_tape_entries WHERE conversation_id=? {clause} ORDER BY id DESC LIMIT ?",
                (self.conversation_id, wanted),
            ).fetchall()
        return [self._row_to_entry(row) for row in reversed(rows)]

    def unresolved(self, limit: int | None = None) -> list[StateTapeEntry]:
        wanted = max(1, int(limit or self.unresolved_entries))
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id,turn_id,source_message_id,status,objective,summary,unresolved_key,resolved,created_at "
                "FROM state_tape_entries WHERE conversation_id=? AND resolved=0 ORDER BY id DESC LIMIT ?",
                (self.conversation_id, wanted),
            ).fetchall()
        return [self._row_to_entry(row) for row in reversed(rows)]

    def _resolve_matching_unresolved(self, objective: str) -> None:
        current_tokens = self._objective_tokens(objective)
        current_key = self._objective_key(objective)
        if not current_tokens and not current_key:
            return
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id,objective,objective_key FROM state_tape_entries "
                "WHERE conversation_id=? AND resolved=0 ORDER BY id DESC LIMIT 8",
                (self.conversation_id,),
            ).fetchall()
            for row_id, previous_objective, previous_key in rows:
                previous_tokens = self._objective_tokens(str(previous_objective or ""))
                union = current_tokens | previous_tokens
                similarity = (len(current_tokens & previous_tokens) / len(union)) if union else 0.0
                if str(previous_key or "") == current_key or similarity >= 0.45:
                    conn.execute(
                        "UPDATE state_tape_entries SET resolved=1,status='resolved',updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (int(row_id),),
                    )

    def commit_turn(
        self,
        *,
        turn_id: int,
        source_message_id: int,
        objective: str,
        assistant_text: str,
        outcomes: Iterable[CompactToolOutcome],
        status: str,
        failure_text: str = "",
    ) -> StateTapeEntry:
        """Commit one protocol-free turn record and trigger deterministic rollup."""

        clean_objective = self._clip(objective, 240)
        rows = list(outcomes)
        blocked = str(status or "").lower() not in {"complete", "resolved"}
        if not blocked:
            self._resolve_matching_unresolved(clean_objective)

        parts: list[str] = []
        for outcome in rows[:4]:
            # Preserve the compact durable observation handle so exact historical
            # evidence can be rehydrated with read_observation without replaying
            # the raw tool payload into every future prompt.
            evidence = (
                f" [observation_id={outcome.observation_id}]"
                if str(outcome.observation_id or "").strip()
                else ""
            )
            parts.append(self._clip(outcome.summary + evidence, 260))
        if failure_text:
            parts.append("Failure: " + self._clip(failure_text, 180))
        if not parts and assistant_text:
            parts.append("Assistant outcome: " + self._clip(assistant_text, 220))
        if not parts:
            parts.append("Turn recorded without a durable tool outcome.")

        prefix = "Unresolved" if blocked else "Completed"
        summary = self._clip(
            f"{prefix} request {clean_objective!r}. " + " ".join(parts),
            self.entry_chars,
        )
        unresolved_key = self._objective_key(clean_objective) if blocked else ""
        objective_key = self._objective_key(clean_objective)

        with _connect() as conn:
            conn.execute(
                """INSERT INTO state_tape_entries(
                       conversation_id,turn_id,source_message_id,status,objective,objective_key,
                       summary,unresolved_key,resolved,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                   ON CONFLICT(conversation_id,turn_id) DO UPDATE SET
                       source_message_id=excluded.source_message_id,
                       status=excluded.status,
                       objective=excluded.objective,
                       objective_key=excluded.objective_key,
                       summary=excluded.summary,
                       unresolved_key=excluded.unresolved_key,
                       resolved=excluded.resolved,
                       updated_at=CURRENT_TIMESTAMP""",
                (
                    self.conversation_id, int(turn_id), max(0, int(source_message_id)),
                    str(status or "complete"), clean_objective, objective_key,
                    summary, unresolved_key, 0 if blocked else 1,
                ),
            )
            row = conn.execute(
                "SELECT id,turn_id,source_message_id,status,objective,summary,unresolved_key,resolved,created_at "
                "FROM state_tape_entries WHERE conversation_id=? AND turn_id=?",
                (self.conversation_id, int(turn_id)),
            ).fetchone()
        self.compact_old_entries()
        assert row is not None
        return self._row_to_entry(row)

    def compact_old_entries(self) -> None:
        """Fold old resolved tape entries into Tier-3 without invoking an LLM."""

        with _connect() as conn:
            rows = conn.execute(
                "SELECT id,turn_id,source_message_id,status,objective,summary,unresolved_key,resolved,created_at "
                "FROM state_tape_entries WHERE conversation_id=? AND resolved=1 ORDER BY id",
                (self.conversation_id,),
            ).fetchall()
        overflow = len(rows) - self.recent_entries
        if overflow <= 0:
            return
        merge_rows = rows[:overflow]
        entries = [self._row_to_entry(row) for row in merge_rows]
        existing = self._single_line(get_conversation_summary(self.conversation_id))
        fragments = [existing] if existing else []
        fragments.extend(entry.render() for entry in entries)
        # Preserve the newest durable facts when the rolling summary reaches its
        # configured bound; state-tape entries themselves remain searchable in
        # raw chat/observation storage.
        combined = self._single_line(" ".join(fragments))
        if len(combined) > self.rolling_summary_chars:
            combined = combined[-self.rolling_summary_chars :].lstrip(" ,;:-")

        through_id = max((entry.source_message_id for entry in entries), default=0)
        if through_id:
            applied = apply_conversation_compaction(
                combined, through_id, conversation_id=self.conversation_id
            )
            if not applied:
                set_conversation_summary(combined, self.conversation_id)
        else:
            set_conversation_summary(combined, self.conversation_id)

        ids = [entry.id for entry in entries]
        placeholders = ",".join("?" for _ in ids)
        with _connect() as conn:
            conn.execute(
                f"DELETE FROM state_tape_entries WHERE conversation_id=? AND id IN ({placeholders})",
                [self.conversation_id, *ids],
            )

    def render_prompt_context(self) -> str:
        """Return compact prompt state with no historical schemas or tool protocol."""

        sections: list[str] = []
        rolling = self._single_line(get_conversation_summary(self.conversation_id))
        if rolling:
            sections.append("Rolling summary:\n" + self._clip(rolling, self.rolling_summary_chars))
        recent = self.recent(self.recent_entries, resolved_only=True)
        if recent:
            sections.append("Recent state tape:\n" + "\n".join(entry.render() for entry in recent))
        unresolved = self.unresolved(self.unresolved_entries)
        if unresolved:
            sections.append(
                "Unresolved work (retain until completed):\n"
                + "\n".join(entry.render() for entry in unresolved)
            )
        return "\n\n".join(sections)


def compact_recent_conversation(
    history: Iterable[dict[str, Any]], *, max_turns: int = 3
) -> list[dict[str, Any]]:
    """Keep only recent user/final-assistant surface text, never tool protocol.

    Raw tool calls, tool results, images, runtime feedback, system messages, and
    empty assistant action rows are intentionally excluded.  This is the Tier-2
    conversational surface used alongside the State Tape.
    """

    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for raw in history:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "")
        if role == "system" or role == "tool" or raw.get("_runtime"):
            continue
        if role == "user":
            if current:
                turns.append(current)
            current = [{"role": "user", "content": str(raw.get("content") or "")}]
            continue
        if role != "assistant" or not current:
            continue
        content = str(raw.get("content") or "").strip()
        if raw.get("tool_calls") or not content:
            continue
        lower = content.lower()
        if "<tool_call" in lower:
            continue
        if lower.startswith("model request failed:") or (
            lower.startswith("turn failed:") and "timed out" in lower
        ):
            continue
        current.append({"role": "assistant", "content": content})
    if current:
        turns.append(current)

    kept = turns[-max(1, int(max_turns)) :]
    result: list[dict[str, Any]] = []
    for turn in kept:
        user = next((item for item in turn if item.get("role") == "user"), None)
        assistant = next(
            (item for item in reversed(turn) if item.get("role") == "assistant"), None
        )
        if user is not None:
            result.append(user)
        if assistant is not None:
            result.append(assistant)
    return result
