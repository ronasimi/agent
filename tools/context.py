"""Bounded conversation context and rolling-summary helpers."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-independent estimate suitable for budgeting local models."""
    if not text:
        return 0
    words = len(str(text).split())
    punctuation = len(re.findall(r"[^\w\s]", str(text), flags=re.UNICODE))
    return max(1, int(words * 1.25 + punctuation * 0.55))


def estimate_messages_tokens(messages: Iterable[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(message.get("content", ""))
        if message.get("tool_calls"):
            total += estimate_tokens(json.dumps(message["tool_calls"], ensure_ascii=False))
    return total


def _truncate_content(content: str, max_tokens: int) -> str:
    """Truncate content so the local token estimator stays within max_tokens."""
    text = str(content)
    if estimate_tokens(text) <= max_tokens:
        return text
    max_tokens = max(1, int(max_tokens))
    suffix = "\n\n[Context truncated by the harness.]"
    if max_tokens <= estimate_tokens(suffix):
        return suffix[: max(1, max_tokens * 3)]

    target = max(1, int(len(text) * max_tokens / max(estimate_tokens(text), 1)))
    lo, hi = 1, min(len(text), max(target * 2, 1))
    best = suffix
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if estimate_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def build_active_messages(
    *,
    system_prompt: str,
    summary: str,
    history: list[dict[str, Any]],
    max_ctx_tokens: int,
    reserve_tokens: int = 2048,
    recent_messages: int = 12,
    extra_prompt_tokens: int = 0,
) -> list[dict[str, Any]]:
    """Construct bounded model context while reserving space for non-message prompt data."""
    base: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if summary:
        bounded_summary = str(summary)[-8000:]
        base.append({
            "role": "system",
            "content": "### Rolling conversation summary\n" + bounded_summary,
        })

    recent = list(history[-max(2, int(recent_messages)):])
    budget = max(128, int(max_ctx_tokens) - int(reserve_tokens) - max(0, int(extra_prompt_tokens)))
    used = estimate_messages_tokens(base)
    selected: list[dict[str, Any]] = []

    for message in reversed(recent):
        candidate_tokens = estimate_messages_tokens([message])
        if used + candidate_tokens <= budget:
            selected.insert(0, message)
            used += candidate_tokens
            continue
        remaining = budget - used
        if remaining < 200:
            break
        copy = dict(message)
        copy["content"] = _truncate_content(str(copy.get("content", "")), max(1, remaining - 64))
        selected.insert(0, copy)
        break
    return base + selected
