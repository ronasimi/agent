"""Tiny-model routing with a reusable catalog prefix and measured request costs."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from .routing_decision import (
    LEARNED_MAX_ADJUSTMENT,
    RankedTool,
    RoutingFeedbackStore,
    _base_score,
    _context_key,
    _schema_parts,
)
from .routing_index import RoutingIndex, build_routing_index

ROUTER_CONFIDENCE = {"H": 0.90, "M": 0.72, "L": 0.48}
DEFAULT_ROUTE_THRESHOLD = 0.62
DEFAULT_CANDIDATES = 8
MAX_REQUEST_CHARS = 700
_ROUTE_RE = re.compile(r"([0-9]{3})\s*([HML])", re.I)


@dataclass(frozen=True)
class RouterDecision:
    selected: tuple[str, ...]
    confidence: float
    tier: str
    candidates: tuple[RankedTool, ...]
    context_key: str
    raw_choice: str = ""
    router_error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    left = max(1, limit // 2 - 2)
    right = max(1, limit - left - 3)
    return text[:left] + "..." + text[-right:]


class CompactCandidateRetriever:
    """Fast prefilter only; it never makes the final routing decision."""

    def __init__(self, feedback: RoutingFeedbackStore | None = None):
        self.feedback = feedback or RoutingFeedbackStore()

    def rank(
        self, text: str, schemas: Iterable[dict], *, limit: int = DEFAULT_CANDIDATES
    ) -> list[RankedTool]:
        key = _context_key(text)
        materialized = list(schemas)
        names = [
            _schema_parts(schema)[0]
            for schema in materialized
            if _schema_parts(schema)[0] not in {"", "tool_search", "load_tools"}
        ]
        learned = self.feedback.score_map(names, key)
        rows: list[RankedTool] = []
        for schema in materialized:
            name, description, _parameters = _schema_parts(schema)
            if not name or name in {"tool_search", "load_tools"}:
                continue
            base = _base_score(text, schema)
            if base <= 0:
                continue
            global_ema, context_ema = learned.get(name, (0.5, 0.5))
            prior = 0.35 * (global_ema - 0.5) + 0.65 * (context_ema - 0.5)
            # Learning may improve shortlist ordering, but cannot manufacture a
            # semantically unrelated candidate.
            adjustment = max(-0.08, min(0.08, prior * 0.16))
            score = max(0.0, min(1.0, base + adjustment))
            rows.append(RankedTool(name, score, base, description, key))
        rows.sort(key=lambda row: (-row.score, row.name))
        return rows[: max(1, min(DEFAULT_CANDIDATES, int(limit)))]


class SystemOneRouter:
    """Jev-like tiny-model router with persistent outcome calibration."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        options: dict[str, Any] | None = None,
        keep_alive: Any = -1,
        candidate_count: int = DEFAULT_CANDIDATES,
        route_threshold: float = DEFAULT_ROUTE_THRESHOLD,
        feedback: RoutingFeedbackStore | None = None,
        inference_slot: Callable[[], Any] | None = None,
        prefix_max_bytes: int = 16000,
        description_chars: int = 48,
        on_metrics: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.client = client
        self.model = str(model)
        self.options = {
            "num_ctx": 8192,
            "temperature": 0,
            "num_predict": 4,
            **(options or {}),
        }
        self.keep_alive = keep_alive
        self.candidate_count = max(2, min(DEFAULT_CANDIDATES, int(candidate_count)))
        self.route_threshold = max(0.0, min(1.0, float(route_threshold)))
        self.feedback = feedback or RoutingFeedbackStore()
        self.retriever = CompactCandidateRetriever(self.feedback)
        self.inference_slot = inference_slot or nullcontext
        self.prefix_max_bytes = int(prefix_max_bytes)
        self.description_chars = int(description_chars)
        self.on_metrics = on_metrics
        self._last_prompt = ""
        self._last_request = ""
        self._last_candidates: tuple[RankedTool, ...] = ()
        self._last_metrics: dict[str, Any] = {}

    def build_index(self, schemas: Iterable[dict]) -> RoutingIndex:
        return build_routing_index(
            schemas,
            num_ctx=int(self.options["num_ctx"]),
            max_prefix_bytes=self.prefix_max_bytes,
            description_chars=self.description_chars,
        )

    def _prompt(
        self, request: str, candidates: list[RankedTool], index: RoutingIndex
    ) -> str:
        ids = index.ids
        choices = ",".join([*(ids[row.name] for row in candidates), "000"])
        return (
            index.prefix
            + "Choices|"
            + choices
            + "\nQ|"
            + _compact_text(request, MAX_REQUEST_CHARS)
            + "\nAnswer|"
        )

    def _generate(self, prompt: str, index: RoutingIndex, *, purpose: str) -> Any:
        started = time.monotonic()
        model_started = None
        response = None
        error = ""
        try:
            with self.inference_slot():
                model_started = time.monotonic()
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options=dict(self.options),
                    keep_alive=self.keep_alive,
                    stream=False,
                )
            return response
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ended = time.monotonic()
            metrics = {
                "wall_ms": round((ended - started) * 1000, 3),
                "queue_wait_ms": round(
                    ((model_started if model_started is not None else ended) - started)
                    * 1000,
                    3,
                ),
                "model_wall_ms": round((ended - model_started) * 1000, 3)
                if model_started is not None
                else None,
                "prefix_fingerprint": index.fingerprint,
                "prefix_bytes": len(index.prefix.encode()),
                "catalog_tools": len(index.names),
                "prompt_chars": len(prompt),
                "dynamic_chars": len(prompt) - len(index.prefix),
            }
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_cached_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            ):
                value = (
                    response.get(key)
                    if isinstance(response, dict)
                    else getattr(response, key, None)
                )
                metrics[key] = value if isinstance(value, (int, float)) else None
            # Missing metrics on older Ollama versions remain null, not zero.
            self._last_metrics = metrics
            if self.on_metrics:
                try:
                    self.on_metrics(
                        {
                            "purpose": purpose,
                            "prompt": prompt,
                            "metrics": metrics,
                            "response": self._response_text(response),
                            "error": error,
                        }
                    )
                except Exception:
                    pass  # Diagnostics must not change the routing outcome.

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, dict):
            return str(response.get("response") or response.get("content") or "")
        return str(
            getattr(response, "response", "") or getattr(response, "content", "")
        )

    def _calibrated_confidence(
        self, tool_name: str, context_key: str, grade: str
    ) -> float:
        base = ROUTER_CONFIDENCE.get(grade.upper(), ROUTER_CONFIDENCE["L"])
        global_ema, context_ema = self.feedback.scores(tool_name, context_key)
        learned = 0.35 * (global_ema - 0.5) + 0.65 * (context_ema - 0.5)
        adjustment = max(
            -LEARNED_MAX_ADJUSTMENT,
            min(LEARNED_MAX_ADJUSTMENT, learned * 2 * LEARNED_MAX_ADJUSTMENT),
        )
        return max(0.0, min(1.0, base + adjustment))

    def decide(
        self,
        text: str,
        schemas: Iterable[dict],
        metadata: dict[str, dict] | None = None,
    ) -> RouterDecision:
        del metadata
        schemas = list(schemas)
        self._last_metrics = {}
        key = _context_key(text)
        candidates = self.retriever.rank(text, schemas, limit=self.candidate_count)
        self._last_request = _compact_text(text, MAX_REQUEST_CHARS)
        self._last_candidates = tuple(candidates)
        if not candidates:
            return RouterDecision((), 0.0, "fallback", (), key)

        try:
            index = self.build_index(schemas)
            prompt = self._prompt(text, candidates, index)
            self._last_prompt = prompt
            response = self._generate(prompt, index, purpose="tool_routing")
            raw = self._response_text(response).strip().upper()
        except Exception as exc:  # fallback remains compact tool_search/load_tools
            return RouterDecision(
                (),
                0.0,
                "fallback",
                tuple(candidates),
                key,
                router_error=f"{type(exc).__name__}: {exc}",
                metrics=dict(self._last_metrics),
            )

        match = _ROUTE_RE.fullmatch(raw)
        if not match:
            return RouterDecision(
                (),
                0.0,
                "fallback",
                tuple(candidates),
                key,
                raw_choice=raw,
                metrics=dict(self._last_metrics),
            )
        choice, grade = match.group(1).upper(), match.group(2).upper()
        if choice == "000":
            return RouterDecision(
                (),
                ROUTER_CONFIDENCE.get(grade, 0.48),
                "none",
                tuple(candidates),
                key,
                raw_choice=raw,
                metrics=dict(self._last_metrics),
            )
        allowed = {index.ids[row.name]: row for row in candidates}
        row = allowed.get(choice)
        if row is None:
            return RouterDecision(
                (),
                0.0,
                "fallback",
                tuple(candidates),
                key,
                raw_choice=raw,
                metrics=dict(self._last_metrics),
            )

        confidence = self._calibrated_confidence(row.name, key, grade)
        selected = (row.name,) if confidence >= self.route_threshold else ()
        tier = "selected" if selected else "fallback"
        return RouterDecision(
            selected,
            confidence,
            tier,
            tuple(candidates),
            key,
            raw_choice=raw,
            metrics=dict(self._last_metrics),
        )

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
            tool_name, context_key, outcome, event_type=event_type, detail=detail
        )

    def reset_turn(self) -> None:
        """Clear all ephemeral routing state; durable calibration stays in SQLite."""
        self._last_prompt = ""
        self._last_request = ""
        self._last_candidates = ()
        self._last_metrics = {}

    def warm(self, schemas: Iterable[dict]) -> bool:
        """Evaluate the real stable index with the same template/options as routing."""
        try:
            index = self.build_index(schemas)
            self._generate(
                self._prompt("No task.", [], index),
                index,
                purpose="router_prefix_warmup",
            )
            self.reset_turn()
            return True
        except Exception:
            self.reset_turn()
            return False

    def compact_candidates(
        self, text: str, schemas: Iterable[dict], limit: int = 6
    ) -> list[dict[str, Any]]:
        rows = self.retriever.rank(
            text, schemas, limit=min(limit, self.candidate_count)
        )
        return [
            {
                "name": row.name,
                "description": _compact_text(row.description, 140),
                "relevance": round(row.score, 3),
            }
            for row in rows
        ]
