"""Reusable deterministic helpers for the interactive turn engine."""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tools import AVAILABLE_TOOLS_MAP, TOOL_METADATA, get_compacted_through_id, get_conversation_summary, get_tool_schema, normalize_arguments, store_tool_observation
from tools.context import build_active_messages, compaction_cutoff_id, estimate_messages_tokens, model_message
from tools.loop_validator import tool_call_signature
from tools.runtime import create_singleton_job
from tools.task_requirements import TaskRequirementLedger, is_followup_request
from tools.conversation_context import get_active_conversation_id
from tools.executor import execute_registered_tool
from .events import emit_event
from .model_protocol import ollama_wire_messages
from .prompts import append_and_save, build_system_prompt
from .state import (
    COMPACT_AT, MAIN_OPTIONS, MAX_CTX, MAX_ITERATIONS, MAX_ITERATIONS_HARD,
    MAX_MUTATING_CALLS_PER_ITERATION, MAX_TOOL_CALLS_PER_ITERATION, MAX_TOOL_OUTPUT,
    MAX_TOOLS_PER_TURN, MODEL, OLLAMA, PRUNE_SATISFIED_REQUIREMENT_TOOLS,
    RECENT_MESSAGES, REQUIREMENT_TOOL_CAP, RESERVE_TOKENS, SUMMARY_KEEP_MESSAGES,
    SUPPRESS_COMPLETED_REQUIREMENT_REPEATS, VOLATILE_CONTEXT_LAST, WORKING_STATE,
    WORKING_STATE_EVIDENCE_CHARS, WORKING_STATE_ENABLED, WORKING_STATE_HISTORY_TURNS,
)

def _parse_tool_calls(raw_calls: Any, allowed_names: set[str] | None = None) -> tuple[list[dict], list[str]]:
    """Normalize native calls and retain actionable parse errors for the model."""
    result: list[dict] = []
    errors: list[str] = []
    allowed = set(allowed_names) if allowed_names is not None else set(AVAILABLE_TOOLS_MAP)
    names_by_lower = {name.lower(): name for name in allowed}
    for index, call in enumerate(raw_calls or [], start=1):
        try:
            if isinstance(call, dict):
                function = call.get("function") or {}
                name = str(function.get("name", "")).strip()
                args = function.get("arguments", {})
                call_id = call.get("id") or uuid.uuid4().hex
            else:
                function = getattr(call, "function", None)
                name = str(getattr(function, "name", "") if function else "").strip()
                args = getattr(function, "arguments", {}) if function else {}
                call_id = getattr(call, "id", None) or uuid.uuid4().hex

            canonical = name if name in allowed else names_by_lower.get(name.lower(), "")
            if not canonical:
                errors.append(f"call {index}: tool '{name or '[missing]'}' was not supplied in this turn")
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as exc:
                    errors.append(f"call {index} ({canonical}): arguments are not valid JSON: {exc.msg}")
                    continue
            if not isinstance(args, dict):
                errors.append(f"call {index} ({canonical}): arguments must be a JSON object")
                continue
            try:
                args = normalize_arguments(AVAILABLE_TOOLS_MAP[canonical], args)
            except Exception as exc:
                errors.append(f"call {index} ({canonical}): invalid arguments: {exc}")
                continue
            result.append({
                "id": call_id,
                "type": "function",
                "function": {"name": canonical, "arguments": args},
            })
        except Exception as exc:
            errors.append(f"call {index}: malformed tool call: {exc}")
    return result, errors




_PROMPT_LEAK_MARKERS = (
    "agent runtime policy",
    "### runtime contract",
    "harness-enforced turn tool policy",
    "the harness working-state block is the authoritative",
    "tools are explicitly typed and supplied through native tool-calling schemas",
)

def _looks_like_prompt_policy_leak(content: str) -> bool:
    """Detect accidental reproduction of hidden harness/system policy text.

    This intentionally looks for distinctive policy phrases rather than generic
    words such as ``system`` or ``tool`` so ordinary discussion is unaffected.
    The check is deterministic and is used before a candidate answer is persisted
    or surfaced as final output.
    """
    text = str(content or "")
    if not text.strip():
        return False
    probe = re.sub(r"[`*_>#]+", " ", text[:2400]).lower()
    probe = re.sub(r"\s+", " ", probe).strip()
    return any(marker.replace("### ", "") in probe for marker in _PROMPT_LEAK_MARKERS)


def _selection_context_for_turn(
    messages: list[dict[str, Any]],
    user_input: str,
    continuation: bool | None = None,
) -> str:
    """Return prior *user* intent only for genuine referential continuations.

    Assistant prose is deliberately excluded.  Otherwise a prior answer that
    casually mentions "system monitoring", "local environment", "web search",
    or another capability can make those unrelated tools appear on the next turn.
    """
    if continuation is None:
        continuation = is_followup_request(user_input)
    if not continuation:
        return ""
    return "\n".join(
        str(item.get("content") or "")
        for item in messages[-7:-1]
        if item.get("role") == "user" and item.get("content")
    )[-1800:]


_FACT_TOOL_INTENTS = {
    "current_time": "current_time",
    "geocode_location": "weather",
    "weather_forecast": "weather",
    "news_search": "news",
    "wiki_search": "encyclopedic",
    "market_quote": "market_price",
}


def _prune_mismatched_fact_tools(
    tool_schemas: list[dict[str, Any]],
    task_frame: dict[str, Any],
    user_input: str,
) -> bool:
    """Remove live-fact primitives selected only by lexical name collision.

    Explicit tool-name requests remain available for debugging/documentation.
    The web discovery primitives are intentionally not pruned because they serve
    many non-news tasks.
    """
    intent = str((task_frame or {}).get("intent") or "")
    raw = str(user_input or "").lower()
    kept: list[dict[str, Any]] = []
    changed = False
    for schema in tool_schemas:
        name = str(schema.get("function", {}).get("name") or "")
        expected = _FACT_TOOL_INTENTS.get(name)
        explicit_name = bool(name and re.search(rf"\b{re.escape(name.lower())}\b", raw))
        if expected and intent != expected and not explicit_name:
            changed = True
            continue
        kept.append(schema)
    if changed:
        tool_schemas[:] = kept
    return changed

def _recover_textual_readonly_tool_call(content: str, allowed_names: set[str]) -> tuple[list[dict], str]:
    """Recover a narrowly formatted read-only pseudo tool call emitted as prose.

    Small local models occasionally print a ``Tool call:`` JSON envelope instead
    of using Ollama's native tool channel.  Only repair this when the text
    explicitly labels itself as a tool call, the JSON names a supplied read-only
    tool, and its arguments validate against the real callable schema. Mutating
    tools are never recovered from prose.
    """
    text = str(content or "")
    if not re.search(r"\btool\s*call\s*:", text, re.I):
        return [], ""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    if not blocks:
        return [], ""
    payload = None
    for raw in reversed(blocks):
        try:
            candidate = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("tool_name"):
            payload = candidate
            break
    if not isinstance(payload, dict):
        return [], ""
    requested = str(payload.get("tool_name") or "").strip()
    canonical = next((name for name in allowed_names if name.lower() == requested.lower()), "")
    if not canonical or canonical not in AVAILABLE_TOOLS_MAP:
        return [], ""
    if not bool(TOOL_METADATA.get(canonical, {}).get("readonly", True)):
        return [], ""

    schema = next((
        item.get("function", {}).get("parameters", {})
        for item in [get_tool_schema(canonical)] if item
    ), {}) or {}
    properties = set((schema.get("properties") or {}).keys())
    args: dict[str, Any] = {}
    nested = payload.get("params")
    if isinstance(nested, dict):
        args.update({k: v for k, v in nested.items() if k in properties})
    args.update({k: v for k, v in payload.items() if k not in {"tool_name", "params"} and k in properties})
    try:
        normalized = normalize_arguments(AVAILABLE_TOOLS_MAP[canonical], args)
    except Exception:
        return [], ""
    return [{
        "id": uuid.uuid4().hex,
        "type": "function",
        "function": {"name": canonical, "arguments": normalized},
    }], canonical


def _extract_tool_calls(raw_calls: Any) -> list[dict]:
    """Compatibility wrapper used by tests/integrations that need valid calls only."""
    calls, _ = _parse_tool_calls(raw_calls)
    return calls

def _sanitize_tool_call_batch(
    calls: list[dict],
    successful_mutating_signatures: set[str],
) -> tuple[list[dict], list[str]]:
    """Bound one 4B-model batch and prevent duplicate mutating side effects."""
    accepted: list[dict] = []
    notes: list[str] = []
    batch_signatures: set[str] = set()
    mutating = 0
    for call in calls:
        name = str(call.get("function", {}).get("name") or "")
        signature = tool_call_signature(call)
        metadata = TOOL_METADATA.get(name, {})
        is_mutating = not bool(metadata.get("readonly", True))
        repeat_safe = bool(metadata.get("repeat_safe", False))
        if signature in batch_signatures:
            notes.append(f"suppressed duplicate call in the same batch: {name}")
            continue
        if is_mutating and not repeat_safe and signature in successful_mutating_signatures:
            notes.append(f"suppressed already-successful duplicate mutating call: {name}")
            continue
        if len(accepted) >= MAX_TOOL_CALLS_PER_ITERATION:
            notes.append(f"suppressed excess call beyond per-iteration limit: {name}")
            continue
        if is_mutating and mutating >= MAX_MUTATING_CALLS_PER_ITERATION:
            notes.append(f"deferred extra mutating call to a later iteration: {name}")
            continue
        accepted.append(call)
        batch_signatures.add(signature)
        if is_mutating:
            mutating += 1
    return accepted, notes

def _tool_status_prefix(success: bool, reason: str, status: str = "") -> str:
    resolved = str(status or ("ok" if success else "error"))
    if resolved == "ok":
        return "[Harness status=ok]"
    safe_reason = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(reason or "tool_error"))[:80]
    if resolved == "partial":
        return f"[Harness status=partial reason={safe_reason}]"
    return f"[Harness status=error reason={safe_reason}]"

def _execute_registered_tool(name: str, args: dict[str, Any]) -> Any:
    """Compatibility wrapper around the canonical tool executor."""
    return execute_registered_tool(name, args)

def _add_recovery_schema(tool_schemas: list[dict], tool_name: str) -> bool:
    """Expose one validator-suggested read-only tool for the corrective iteration.

    Mutating tools must have been selected from the user's original request; an
    untrusted tool transcript cannot indirectly expand the side-effect surface.
    """
    name = str(tool_name or "")
    if not name:
        return False
    if any(str(schema.get("function", {}).get("name") or "") == name for schema in tool_schemas):
        return False
    if not bool(TOOL_METADATA.get(name, {}).get("readonly", True)):
        return False
    schema = get_tool_schema(name)
    if not schema:
        return False
    tool_schemas.append(schema)
    return True

def _ensure_tool_schemas(tool_schemas: list[dict], tool_names: list[str], policy) -> list[str]:
    """Expose explicitly required tool schemas, respecting harness policy.

    Unlike validator recovery, this may include an explicitly requested safe
    artifact tool such as take_web_screenshot.  It never invents schemas and
    reports names that remain blocked/unavailable.
    """
    current = {str(schema.get("function", {}).get("name") or "") for schema in tool_schemas}
    blocked: list[str] = []
    for name in tool_names:
        if name in current:
            continue
        metadata = TOOL_METADATA.get(name, {})
        if name not in AVAILABLE_TOOLS_MAP or not policy.allowed(name, metadata):
            blocked.append(name)
            continue
        schema = get_tool_schema(name)
        if not schema:
            blocked.append(name)
            continue
        tool_schemas.append(schema)
        current.add(name)
    # Bound optional schema growth without ever dropping explicit requirements.
    target_cap = min(REQUIREMENT_TOOL_CAP, max(MAX_TOOLS_PER_TURN, len(set(tool_names)) + 2))
    if len(tool_schemas) > target_cap:
        required = set(tool_names)
        kept = [schema for schema in tool_schemas if str(schema.get("function", {}).get("name") or "") in required]
        for schema in tool_schemas:
            if len(kept) >= target_cap:
                break
            name = str(schema.get("function", {}).get("name") or "")
            if name not in required:
                kept.append(schema)
        tool_schemas[:] = kept
    return blocked

def _adaptive_iteration_limit(requirement_count: int) -> int:
    """Give broad explicit multi-check requests enough room without unbounded loops."""
    # Broad diagnostic prompts commonly need one call per explicit requirement,
    # plus a few recovery/finalization turns.  Keep a hard cap, but do not make
    # one transient parser/tool failure consume the entire completion budget.
    needed = max(MAX_ITERATIONS, int(requirement_count) + 10)
    return min(MAX_ITERATIONS_HARD, needed)

def _user_requests_recheck(user_text: str) -> bool:
    """Return True when repeated observations are explicitly part of the task."""
    lower = " ".join(str(user_text or "").lower().split())
    phrases = (
        "recheck", "re-check", "check again", "run again", "repeat the", "refresh",
        "monitor", "over time", "compare before", "compare after", "watch for",
        "take another", "second snapshot", "again after",
    )
    return any(phrase in lower for phrase in phrases)

def _suppress_completed_requirement_calls(
    calls: list[dict[str, Any]],
    requirement_ledger: TaskRequirementLedger,
    successful_readonly_signatures: set[str],
    user_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Suppress exact successful repeats of completed explicit checks.

    Small models often revisit a familiar successful snapshot instead of moving
    to the next pending requirement.  Rechecks remain available when the user
    explicitly asks for monitoring/change detection.
    """
    if not SUPPRESS_COMPLETED_REQUIREMENT_REPEATS or _user_requests_recheck(user_text):
        return calls, []
    accepted: list[dict[str, Any]] = []
    notes: list[str] = []
    for call in calls:
        name = str(call.get("function", {}).get("name") or "")
        signature = tool_call_signature(call)
        if (
            signature in successful_readonly_signatures
            and requirement_ledger.status_for_tool(name) in {"satisfied", "partial"}
        ):
            notes.append(f"suppressed redundant completed requirement call: {name}")
            continue
        accepted.append(call)
    return accepted, notes

def _schema_additions_pending(tool_schemas: list[dict[str, Any]], pending_names: list[str]) -> bool:
    """Return whether any pending requirement tool is missing from the set."""
    current = {str(schema.get("function", {}).get("name") or "") for schema in tool_schemas}
    return any(name not in current for name in pending_names)

def _refresh_requirement_tool_schemas(
    tool_schemas: list[dict[str, Any]],
    requirement_ledger: TaskRequirementLedger,
    turn_tool_policy,
    *,
    prune: bool = True,
    minimize_churn: bool = False,
) -> bool:
    """Prune completed requirement schemas and prioritize remaining checks.

    Native tool schemas are a substantial part of Ollama prefill.  Once a
    required check is complete, keeping that schema in every later request is
    unnecessary and also invites a 4B model to repeat the familiar call.

    None of that is free, though: chat templates render the tool schemas at the
    very top of the prompt, so any change to the set *or to its order*
    invalidates the whole server-side prefix cache and the next request pays a
    full re-prefill.  ``prune=False`` lets a caller keep the set untouched.

    ``minimize_churn`` is the setting used by the interactive loop: satisfied
    schemas are dropped, and the pending-first ordering is applied, only in an
    iteration that has to change the set anyway to expose a still-pending
    requirement.  Otherwise the serialized set stays byte-identical between
    iterations.  The behaviors this trades away are already covered elsewhere:
    repeats of a completed check are suppressed by
    ``_suppress_completed_requirement_calls``, and the pending requirement list
    is carried explicitly by the harness working state.
    """
    changed = False
    pending_names = requirement_ledger.required_tools(pending_only=True)
    additions_pending = _schema_additions_pending(tool_schemas, pending_names)
    if prune and PRUNE_SATISFIED_REQUIREMENT_TOOLS:
        closed = requirement_ledger.closed_tools()
        if closed and (not minimize_churn or additions_pending):
            before = len(tool_schemas)
            tool_schemas[:] = [
                schema for schema in tool_schemas
                if str(schema.get("function", {}).get("name") or "") not in closed
            ]
            changed = len(tool_schemas) != before

    before_ensure = len(tool_schemas)
    blocked = _ensure_tool_schemas(tool_schemas, pending_names, turn_tool_policy)
    if len(tool_schemas) != before_ensure:
        changed = True
    for name in blocked:
        requirement_ledger.mark_blocked(name, "blocked or unavailable under harness policy")

    # Pending requirements first helps small models pick the next unfinished
    # check and keeps ordering deterministic. Reordering an otherwise unchanged
    # set costs exactly as much prefill as changing it, so under minimize_churn
    # it only happens when the set changed in this call.
    if changed or not minimize_churn:
        pending_order = {name: idx for idx, name in enumerate(requirement_ledger.required_tools(pending_only=True))}
        original_order = {id(schema): idx for idx, schema in enumerate(tool_schemas)}
        tool_schemas.sort(
            key=lambda schema: (
                0 if str(schema.get("function", {}).get("name") or "") in pending_order else 1,
                pending_order.get(str(schema.get("function", {}).get("name") or ""), 10_000),
                original_order.get(id(schema), 10_000),
            )
        )
    return changed or bool(blocked)

def _prune_compacted_history(messages: list[dict]) -> None:
    """Drop rows already represented by the durable rolling summary."""
    watermark = get_compacted_through_id()
    if not watermark:
        return
    messages[1:] = [message for message in messages[1:] if int(message.get("_db_id") or 0) > watermark]

def _queue_compaction_if_needed(messages: list[dict]) -> bool:
    """Queue low-priority compaction after the answer leaves the foreground path."""
    history = messages[1:]
    if estimate_messages_tokens(history) < COMPACT_AT:
        return False
    through_id = compaction_cutoff_id(history, SUMMARY_KEEP_MESSAGES)
    if not through_id:
        return False
    return bool(create_singleton_job(
        "context_compaction",
        "Compact conversation context",
        payload={"through_id": through_id, "conversation_id": get_active_conversation_id()},
        priority=-10,
        max_attempts=5,
        singleton_key=get_active_conversation_id(),
    ))

def _bounded_tool_result_with_ref(tool_name: str, result: Any) -> tuple[str, str]:
    """Keep a head/tail preview and return the durable observation handle."""
    text = str(result)
    # Persist moderately large successful evidence even when it fits in the
    # current context window. Follow-up turns can then retrieve the exact prior
    # observation instead of re-running a web/system tool.
    observation_id = store_tool_observation(tool_name, text) if len(text) >= min(1000, MAX_TOOL_OUTPUT) else ""
    if len(text) <= MAX_TOOL_OUTPUT:
        return text, observation_id
    if not observation_id:
        observation_id = store_tool_observation(tool_name, text)
    marker = (
        f"\n\n[Harness: middle truncated; full {len(text)}-character result stored as observation "
        f"{observation_id}. Use read_observation(observation_id, offset, length) for another slice.]\n\n"
    )
    remaining = max(200, MAX_TOOL_OUTPUT - len(marker))
    head = remaining // 2
    return text[:head].rstrip() + marker + text[-(remaining - head):].lstrip(), observation_id

def _bounded_tool_result(tool_name: str, result: Any) -> str:
    """Compatibility wrapper returning only the bounded prompt text."""
    return _bounded_tool_result_with_ref(tool_name, result)[0]

def _finalize_after_limit(
    messages: list[dict],
    turn_tail: list[dict[str, Any]] | None = None,
    reason: str = "The tool-call safety limit was reached.",
    recovery_context: str = "",
) -> None:
    """Produce a bounded no-tools final answer while preserving latest media."""
    history = messages[1:]
    if WORKING_STATE_ENABLED:
        last_user = next((model_message(item) for item in reversed(history) if item.get("role") == "user"), None)
        history = [last_user] if last_user else []
    prompt = [
        {"role": "system", "content": build_system_prompt()},
        *build_active_messages(
            system_prompt="",
            summary="" if WORKING_STATE_ENABLED else get_conversation_summary(),
            history=history,
            max_ctx_tokens=MAX_CTX,
            reserve_tokens=RESERVE_TOKENS,
            recent_messages=RECENT_MESSAGES,
            extra_prompt_tokens=0,
            working_state=WORKING_STATE.render(include_tool_capabilities=False) if WORKING_STATE_ENABLED else "",
            evidence_context=WORKING_STATE.render_evidence(WORKING_STATE_EVIDENCE_CHARS) if WORKING_STATE_ENABLED else "",
            max_history_turns=WORKING_STATE_HISTORY_TURNS if WORKING_STATE_ENABLED else None,
            volatile_last=VOLATILE_CONTEXT_LAST,
        )[1:],
    ]
    latest_media = next(
        (model_message(item) for item in reversed(turn_tail or []) if item.get("images")),
        None,
    )
    if latest_media:
        prompt.append(latest_media)
    if recovery_context:
        prompt.append({
            "role": "user",
            "content": str(recovery_context)[:6000],
        })
    prompt.append({
        "role": "user",
        "content": (
            f"{str(reason).strip()} Summarize what has been established, what remains incomplete, "
            "and any useful next steps. Do not call tools. If media is attached, ground visual claims only in those pixels."
        ),
    })
    try:
        # Streamed so the user sees the first token of the fallback summary
        # immediately instead of waiting for the whole answer to be generated.
        response = OLLAMA.chat(
            model=MODEL, messages=ollama_wire_messages(prompt), options=MAIN_OPTIONS,
            tools=[], think=False, keep_alive=-1, stream=True,
        )
        # A client that ignores ``stream`` returns one complete response object.
        chunks = [response] if isinstance(response, dict) or hasattr(response, "message") else response
        content = ""
        for chunk in chunks:
            chunk_msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
            piece = chunk_msg.get("content", "") if isinstance(chunk_msg, dict) else getattr(chunk_msg, "content", "")
            if not piece:
                continue
            if not content:
                print("\nAgent: ", end="", flush=True)
            content += piece
            print(piece, end="", flush=True)
            emit_event("assistant_delta", content=piece)
        if content:
            print()
            append_and_save(messages, {"role": "assistant", "content": content})
            emit_event("assistant_final", content=content, finalization=True)
    except Exception as exc:
        print(f"  \033[91m[!] Finalization failed: {exc}\033[0m")
