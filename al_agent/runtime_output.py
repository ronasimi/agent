"""Container-log helpers for the Web UI runtime."""
from __future__ import annotations

import logging
from typing import Any

from .state import SHOW_PERF_STATS

LOGGER = logging.getLogger("al_agent.runtime")


class OperationStatus:
    """Log the lifetime of an internal operation without terminal animation."""

    def __init__(self, message: str = "Processing"):
        self.message = str(message).replace("\n", " ").replace("\r", " ")[:120]

    def __enter__(self):
        LOGGER.debug("%s started", self.message)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            LOGGER.debug("%s completed", self.message)
        else:
            LOGGER.warning("%s failed: %s", self.message, exc_val)


def log_perf_stats(stats: dict[str, Any]) -> None:
    if not SHOW_PERF_STATS or not stats.get("done"):
        return
    prompt_count = stats.get("prompt_eval_count")
    cached_count = stats.get("prompt_eval_cached_count")
    prompt_ns = stats.get("prompt_eval_duration")
    eval_count = stats.get("eval_count")
    eval_ns = stats.get("eval_duration")
    load_ns = stats.get("load_duration")
    ttft_ms = stats.get("_ttft_ms")
    if prompt_count is None and eval_count is None:
        return
    prompt_ms = (float(prompt_ns) / 1_000_000.0) if prompt_ns else 0.0
    eval_ms = (float(eval_ns) / 1_000_000.0) if eval_ns else 0.0
    cached = int(cached_count or 0)
    uncached = max(0, int(prompt_count or 0) - cached)
    cached_pct = (100.0 * cached / int(prompt_count)) if prompt_count else 0.0
    prefill_rate = (uncached / (prompt_ms / 1000.0)) if prompt_ms and uncached else 0.0
    generation_rate = (int(eval_count) / (eval_ms / 1000.0)) if eval_ms and eval_count else 0.0
    fields = []
    if ttft_ms is not None:
        fields.append(f"TTFT {float(ttft_ms):.0f} ms")
    if prompt_count is not None:
        fields.append(
            f"prompt {prompt_count}; cached {cached} ({cached_pct:.1f}%); "
            f"uncached {uncached}; prefill {prefill_rate:.1f} tok/s"
        )
    if eval_count is not None:
        fields.append(f"generation {eval_count} in {eval_ms:.0f} ms ({generation_rate:.1f} tok/s)")
    if load_ns is not None:
        fields.append(f"load {float(load_ns) / 1_000_000.0:.0f} ms")
    LOGGER.info("Ollama: %s", "; ".join(fields))
