"""Token-budgeted, turn-aware conversation context helpers."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

_MESSAGE_FIELDS = {"role", "content", "name", "tool_calls", "tool_call_id", "images"}
IMAGE_TOKEN_ESTIMATE = 1200


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
        if message.get("images"):
            # Vision encoders add image tokens that are not represented by the
            # base64/string length. Reserve a conservative fixed budget per image.
            total += IMAGE_TOKEN_ESTIMATE * len(message["images"])
    return total


def model_message(message: dict[str, Any]) -> dict[str, Any]:
    """Strip local bookkeeping fields before sending a message to Ollama."""
    return {key: value for key, value in message.items() if key in _MESSAGE_FIELDS}


def split_turns(history: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group history into whole user turns, preserving tool-call/result pairs."""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for raw in history:
        message = model_message(raw)
        if message.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def _truncate_content(content: str, max_tokens: int, *, head_tail: bool = False) -> str:
    """Truncate content while keeping the local estimate within max_tokens."""
    text = str(content)
    if estimate_tokens(text) <= max_tokens:
        return text
    max_tokens = max(1, int(max_tokens))
    marker = "\n\n[Context truncated by the harness.]\n\n"
    if max_tokens <= estimate_tokens(marker):
        return marker[: max(1, max_tokens * 3)]

    target_chars = max(1, int(len(text) * max_tokens / max(estimate_tokens(text), 1)))
    lo, hi = 1, min(len(text), max(target_chars * 2, 1))
    best = marker.strip()
    while lo <= hi:
        count = (lo + hi) // 2
        if head_tail:
            head = count // 2
            candidate = text[:head].rstrip() + marker + text[-(count - head):].lstrip()
        else:
            candidate = text[:count].rstrip() + marker.rstrip()
        if estimate_tokens(candidate) <= max_tokens:
            best = candidate
            lo = count + 1
        else:
            hi = count - 1
    return best


def _fit_latest_turn(turn: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Fit the current turn, trimming observations before assistant/user prose."""
    fitted = [dict(message) for message in turn]
    if estimate_messages_tokens(fitted) <= budget:
        return fitted

    tool_indexes = [i for i, message in enumerate(fitted) if message.get("role") == "tool"]
    for index in tool_indexes:
        if estimate_messages_tokens(fitted) <= budget:
            break
        content = str(fitted[index].get("content", ""))
        fitted[index]["content"] = _truncate_content(content, min(512, max(96, budget // 4)), head_tail=True)

    for index, message in enumerate(fitted):
        if estimate_messages_tokens(fitted) <= budget:
            break
        if message.get("role") == "assistant" and message.get("content"):
            fitted[index]["content"] = _truncate_content(str(message["content"]), min(256, max(64, budget // 6)))

    while len(fitted) > 1 and estimate_messages_tokens(fitted) > budget:
        removable = next(
            (i for i, message in enumerate(fitted[:-1]) if message.get("role") in {"assistant", "tool"}),
            None,
        )
        if removable is None:
            break
        fitted.pop(removable)

    if estimate_messages_tokens(fitted) > budget:
        user_index = next((i for i in range(len(fitted) - 1, -1, -1) if fitted[i].get("role") == "user"), len(fitted) - 1)
        fitted[user_index]["content"] = _truncate_content(str(fitted[user_index].get("content", "")), max(64, budget // 2), head_tail=True)
    return fitted


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
    """Build bounded context from whole turns, always preserving the current turn."""
    del recent_messages  # retained for configuration/API compatibility
    base: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if summary:
        base.append({
            "role": "system",
            "content": "### Rolling conversation summary\n" + _truncate_content(str(summary), 2000, head_tail=True),
        })

    budget = max(128, int(max_ctx_tokens) - int(reserve_tokens) - max(0, int(extra_prompt_tokens)))
    used = estimate_messages_tokens(base)
    available = max(64, budget - used)
    turns = split_turns(history)
    selected: list[list[dict[str, Any]]] = []

    for reverse_index, turn in enumerate(reversed(turns)):
        turn_tokens = estimate_messages_tokens(turn)
        if turn_tokens <= available:
            selected.insert(0, turn)
            available -= turn_tokens
            continue
        if reverse_index == 0:
            selected.insert(0, _fit_latest_turn(turn, available))
        break

    return base + [message for turn in selected for message in turn]


def fit_tool_loop_messages(
    prefix: list[dict[str, Any]],
    tail: list[dict[str, Any]],
    *,
    max_ctx_tokens: int,
    reserve_tokens: int,
    extra_prompt_tokens: int = 0,
) -> list[dict[str, Any]]:
    """Preserve a frozen prefix and bound a tool-loop suffix only when necessary."""
    budget = max(128, int(max_ctx_tokens) - int(reserve_tokens) - max(0, int(extra_prompt_tokens)))
    fitted = [model_message(message) for message in tail]
    if estimate_messages_tokens(prefix) + estimate_messages_tokens(fitted) <= budget:
        return [*prefix, *fitted]

    for index, message in enumerate(fitted):
        if estimate_messages_tokens(prefix) + estimate_messages_tokens(fitted) <= budget:
            break
        if message.get("role") == "tool":
            fitted[index]["content"] = _truncate_content(str(message.get("content", "")), 256, head_tail=True)

    for index, message in enumerate(fitted):
        if estimate_messages_tokens(prefix) + estimate_messages_tokens(fitted) <= budget:
            break
        if message.get("role") == "assistant" and message.get("content"):
            fitted[index]["content"] = _truncate_content(str(message["content"]), 128)

    while estimate_messages_tokens(prefix) + estimate_messages_tokens(fitted) > budget:
        assistant_indexes = [i for i, message in enumerate(fitted) if message.get("role") == "assistant"]
        if len(assistant_indexes) <= 1:
            break
        fitted = fitted[assistant_indexes[1]:]

    return [*prefix, *fitted]


def compaction_cutoff_id(history: list[dict[str, Any]], keep_messages: int = 8) -> int:
    """Return the last DB id that can be summarized while keeping recent whole turns."""
    if len(history) <= max(2, int(keep_messages)):
        return 0
    target = max(1, len(history) - max(2, int(keep_messages)))
    while target > 0 and history[target].get("role") != "user":
        target -= 1
    if target <= 0 or target >= len(history):
        return 0
    ids = [int(message.get("_db_id") or 0) for message in history[:target]]
    return max(ids, default=0)
