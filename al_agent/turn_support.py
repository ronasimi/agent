"""Reusable deterministic helpers for the interactive turn engine."""
from __future__ import annotations

import json
import re
import signal
import threading
import uuid
from typing import Any

from tools import AVAILABLE_TOOLS_MAP, TOOL_METADATA, get_compacted_through_id, get_conversation_summary, get_tool_schema, normalize_arguments, store_tool_observation
from tools.context import build_active_messages, compaction_cutoff_id, estimate_messages_tokens, model_message
from tools.loop_validator import tool_call_signature
from tools.runtime import create_singleton_job
from tools.task_requirements import TaskRequirementLedger
from .events import emit_event
from .prompts import append_and_save, build_system_prompt
from .state import (
    COMPACT_AT, MAIN_OPTIONS, MAX_CTX, MAX_ITERATIONS, MAX_ITERATIONS_HARD,
    MAX_MUTATING_CALLS_PER_ITERATION, MAX_TOOL_CALLS_PER_ITERATION, MAX_TOOL_OUTPUT,
    MAX_TOOLS_PER_TURN, MODEL, OLLAMA, PRUNE_SATISFIED_REQUIREMENT_TOOLS,
    RECENT_MESSAGES, REQUIREMENT_TOOL_CAP, RESERVE_TOKENS, SUMMARY_KEEP_MESSAGES,
    SUPPRESS_COMPLETED_REQUIREMENT_REPEATS, WORKING_STATE, WORKING_STATE_EVIDENCE_CHARS,
    WORKING_STATE_ENABLED, WORKING_STATE_HISTORY_TURNS,
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
    """Execute one tool, enforcing decorator timeouts for custom tools.

    Built-ins already implement operation-specific subprocess/network timeouts.
    Custom tools default to a 60-second SIGALRM guard so a buggy generated tool
    cannot wedge the interactive loop indefinitely.
    """
    func = AVAILABLE_TOOLS_MAP[name]
    timeout = TOOL_METADATA.get(name, {}).get("timeout")
    if not timeout or threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        return func(**args)
    seconds = max(1, min(int(timeout), 300))
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"Tool '{name}' exceeded its {seconds}-second harness timeout.")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return func(**args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

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
    """Suppress exact successful repeats when other explicit checks are pending.

    Small models often revisit a familiar successful snapshot instead of moving
    to the next pending requirement.  Rechecks remain available when the user
    explicitly asks for monitoring/change detection.
    """
    if not SUPPRESS_COMPLETED_REQUIREMENT_REPEATS or not requirement_ledger.pending() or _user_requests_recheck(user_text):
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

def _refresh_requirement_tool_schemas(
    tool_schemas: list[dict[str, Any]],
    requirement_ledger: TaskRequirementLedger,
    turn_tool_policy,
) -> bool:
    """Prune completed requirement schemas and prioritize remaining checks.

    Native tool schemas are a substantial part of Ollama prefill.  Once a
    required check is complete, keeping that schema in every later request is
    unnecessary and also invites a 4B model to repeat the familiar call.
    """
    changed = False
    if PRUNE_SATISFIED_REQUIREMENT_TOOLS:
        closed = requirement_ledger.closed_tools()
        if closed:
            before = len(tool_schemas)
            tool_schemas[:] = [
                schema for schema in tool_schemas
                if str(schema.get("function", {}).get("name") or "") not in closed
            ]
            changed = len(tool_schemas) != before

    pending_names = requirement_ledger.required_tools(pending_only=True)
    before_ensure = len(tool_schemas)
    blocked = _ensure_tool_schemas(tool_schemas, pending_names, turn_tool_policy)
    if len(tool_schemas) != before_ensure:
        changed = True
    for name in blocked:
        requirement_ledger.mark_blocked(name, "blocked or unavailable under harness policy")

    # Pending requirements first helps small models pick the next unfinished
    # check and keeps ordering deterministic for prompt-cache friendliness.
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
        payload={"through_id": through_id},
        priority=-10,
        max_attempts=5,
    ))

def _bounded_tool_result_with_ref(tool_name: str, result: Any) -> tuple[str, str]:
    """Keep a head/tail preview and return the durable observation handle."""
    text = str(result)
    if len(text) <= MAX_TOOL_OUTPUT:
        return text, ""
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
        )[1:],
    ]
    latest_media = next(
        (model_message(item) for item in reversed(turn_tail or []) if item.get("images")),
        None,
    )
    if latest_media:
        prompt.append(latest_media)
    prompt.append({
        "role": "user",
        "content": (
            f"{str(reason).strip()} Summarize what has been established, what remains incomplete, "
            "and any useful next steps. Do not call tools. If media is attached, ground visual claims only in those pixels."
        ),
    })
    try:
        response = OLLAMA.chat(model=MODEL, messages=prompt, options=MAIN_OPTIONS, tools=[], think=False, keep_alive=-1)
        msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if content:
            print(f"\nAgent: {content}\n")
            append_and_save(messages, {"role": "assistant", "content": content})
            emit_event("assistant_final", content=content, finalization=True)
    except Exception as exc:
        print(f"  \033[91m[!] Finalization failed: {exc}\033[0m")
