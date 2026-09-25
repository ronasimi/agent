"""Non-generative tool candidate retrieval for the resident-model runtime.

This module deliberately contains no Ollama/client dependency. It narrows the
catalog cheaply, then lets the already-resident main model make the semantic
decision and emit the tool call.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from .routing_decision import (
    RankedTool,
    RoutingDecision,
    RoutingFeedbackStore,
    _base_score,
    _context_key,
    _schema_parts,
    _tokens,
)

DEFAULT_CANDIDATES = 8
DEFAULT_AUTO_ACTIVATE_THRESHOLD = 0.80
DEFAULT_AUTO_ACTIVATE_MARGIN = 0.20
DEFAULT_MIN_CANDIDATE_SCORE = 0.18


def _request_segments(text: str) -> list[str]:
    """Extract primary task clauses so formatting constraints do not dominate routing.

    Numbered prompts are treated as explicit independent requirements. For each
    numbered item only the first sentence is used for catalog retrieval; trailing
    sentences commonly describe output/source constraints rather than new tools.
    """
    raw_text = str(text or "")
    numbered: list[str] = []
    for line in raw_text.splitlines():
        match = re.match(r"^\s*\d+[.)]\s*(.+)$", line)
        if not match:
            continue
        primary = re.split(r"[.!?;]+", match.group(1), maxsplit=1)[0].strip()
        if primary:
            numbered.append(primary)
    if numbered:
        return numbered

    segments: list[str] = []
    for raw in re.split(r"[\n.!?;]+", raw_text):
        cleaned = raw.strip()
        if cleaned:
            segments.append(cleaned)
    return segments or [raw_text]


class DeterministicToolRouter:
    """Rank and activate a small relevant schema set without another model call."""

    def __init__(
        self,
        *,
        candidate_count: int = DEFAULT_CANDIDATES,
        auto_activate_threshold: float = DEFAULT_AUTO_ACTIVATE_THRESHOLD,
        auto_activate_margin: float = DEFAULT_AUTO_ACTIVATE_MARGIN,
        min_candidate_score: float = DEFAULT_MIN_CANDIDATE_SCORE,
        feedback: RoutingFeedbackStore | None = None,
    ) -> None:
        self.candidate_count = max(1, min(8, int(candidate_count)))
        self.auto_activate_threshold = max(0.0, min(1.0, float(auto_activate_threshold)))
        self.auto_activate_margin = max(0.0, min(1.0, float(auto_activate_margin)))
        self.min_candidate_score = max(0.0, min(1.0, float(min_candidate_score)))
        self.feedback = feedback or RoutingFeedbackStore()

    @staticmethod
    def _name_idf(schemas: list[dict]) -> tuple[dict[str, float], float]:
        names = []
        df: Counter[str] = Counter()
        for schema in schemas:
            name, _description, _parameters = _schema_parts(schema)
            if not name or name in {"tool_search", "load_tools"}:
                continue
            terms = set(_tokens(name.replace("_", " ")))
            names.append(terms)
            df.update(terms)
        total = max(1, len(names))
        idf = {
            term: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }
        return idf, max(idf.values(), default=1.0)

    def rank(
        self,
        text: str,
        schemas: Iterable[dict],
        *,
        limit: int | None = None,
    ) -> list[RankedTool]:
        key = _context_key(text)
        materialized = list(schemas)
        segments = _request_segments(text)
        names = [
            _schema_parts(schema)[0]
            for schema in materialized
            if _schema_parts(schema)[0] not in {"", "tool_search", "load_tools"}
        ]
        learned = self.feedback.score_map(names, key)
        name_idf, max_name_idf = self._name_idf(materialized)
        rows_by_segment: list[list[RankedTool]] = []
        best_by_name: dict[str, RankedTool] = {}

        for segment in segments:
            q_terms = set(_tokens(segment))
            rows: list[RankedTool] = []
            for schema in materialized:
                name, description, _parameters = _schema_parts(schema)
                if not name or name in {"tool_search", "load_tools"}:
                    continue
                base = _base_score(segment, schema)
                if base <= 0:
                    continue
                name_terms = set(_tokens(name.replace("_", " ")))
                matched_name_weight = sum(name_idf.get(t, 0.0) for t in q_terms & name_terms)
                # Rare exact name-token matches disambiguate generic verbs such
                # as "host"/"read" without hard-coding individual tools.
                specificity_bonus = min(0.10, 0.10 * matched_name_weight / max_name_idf)
                global_ema, context_ema = learned.get(name, (0.5, 0.5))
                prior = 0.35 * (global_ema - 0.5) + 0.65 * (context_ema - 0.5)
                adjustment = max(-0.08, min(0.08, prior * 0.16))
                score = max(0.0, min(1.0, base + specificity_bonus + adjustment))
                row = RankedTool(name, score, base, description, key)
                rows.append(row)
                previous = best_by_name.get(name)
                if previous is None or row.score > previous.score:
                    best_by_name[name] = row
            rows.sort(key=lambda row: (-row.score, row.name))
            rows_by_segment.append(rows)

        cap = self.candidate_count if limit is None else max(1, min(8, int(limit)))
        if not best_by_name:
            return []

        # Multi-requirement prompts must retain coverage across independent
        # clauses instead of allowing one verbose clause to occupy the shortlist.
        ordered: list[RankedTool] = []
        seen: set[str] = set()
        if len(rows_by_segment) > 1:
            for rows in rows_by_segment:
                if not rows:
                    continue
                winner = rows[0]
                if winner.name not in seen:
                    ordered.append(best_by_name[winner.name])
                    seen.add(winner.name)
                    if len(ordered) >= cap:
                        return ordered

        remaining = sorted(best_by_name.values(), key=lambda row: (-row.score, row.name))
        for row in remaining:
            if row.name in seen:
                continue
            ordered.append(row)
            seen.add(row.name)
            if len(ordered) >= cap:
                break
        return ordered

    def decide(
        self,
        text: str,
        schemas: Iterable[dict],
        metadata: dict[str, dict] | None = None,
    ) -> RoutingDecision:
        del metadata
        key = _context_key(text)
        candidates = tuple(self.rank(text, schemas, limit=self.candidate_count))
        if not candidates:
            return RoutingDecision((), 0.0, "fallback", (), key)

        top_by_score = sorted(candidates, key=lambda row: (-row.score, row.name))
        top = top_by_score[0]
        runner_up = top_by_score[1].score if len(top_by_score) > 1 else 0.0
        margin = max(0.0, top.score - runner_up)
        single_intent = len(_request_segments(text)) == 1
        if (
            single_intent
            and top.score >= self.auto_activate_threshold
            and margin >= self.auto_activate_margin
        ):
            selected = (top.name,)
            tier = "direct"
        else:
            selected = tuple(
                row.name for row in candidates if row.score >= self.min_candidate_score
            )
            tier = "candidate_set" if selected else "fallback"
        return RoutingDecision(selected, top.score, tier, candidates, key)

    def record(
        self,
        tool_name: str,
        context_key: str,
        outcome: float | None,
        *,
        event_type: str,
        detail: str = "",
    ) -> None:
        self.feedback.record(
            tool_name,
            context_key,
            outcome,
            event_type=event_type,
            detail=detail,
        )

    def reset_turn(self) -> None:
        """Compatibility no-op; deterministic routing has no per-turn model state."""
        return None
