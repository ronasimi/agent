# ==========================================
# FILE: agent.py
# ==========================================
"""Interactive frontend for the local agent runtime.

Long-running work is intentionally delegated to worker.py and persisted in
SQLite; the CLI process can exit without terminating background jobs.
"""
from __future__ import annotations

import base64
import itertools
import json
import os
import re
import signal
import sys
import threading
import time
import uuid
from io import BytesIO
from typing import Any

from ollama import Client
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from tools.config import load_config

try:
    from prompt_toolkit.auto_suggestion import AutoSuggestFromHistory
    AUTO_SUGGEST = AutoSuggestFromHistory()
except ImportError:  # pragma: no cover
    AUTO_SUGGEST = None

from tools import (
    AVAILABLE_TOOLS_MAP,
    TOOL_METADATA,
    _load_chat_history_from_db,
    _save_message_to_db,
    clear_chat_history,
    get_compacted_through_id,
    get_conversation_summary,
    get_relevant_memories,
    get_tools_prompt_summary,
    get_tool_schema,
    init_db,
    load_tools,
    normalize_arguments,
    search_memory,
    select_tool_schemas,
    store_tool_observation,
)
from tools.context import (
    build_active_messages,
    compaction_cutoff_id,
    estimate_messages_tokens,
    estimate_tokens,
    fit_tool_loop_messages,
    model_message,
)
from tools.job_tools import enqueue_research, get_research_status
from tools.model_context import SharedModelContext
from tools.media import unpack_media_result
from tools.loop_validator import (
    StepFailureTracker,
    build_recovery_message,
    build_stall_recovery_message,
    classify_tool_outcome,
    select_recovery_tool_calls,
    select_stall_recovery_tool_calls,
    tool_call_signature,
    validate_stalled_step,
    validate_tool_loop,
)
from tools.runtime import create_singleton_job, list_jobs, record_monitor_state, utc_now
from tools.self_optimization import (
    approve_self_optimization,
    enqueue_self_optimization,
    list_self_optimization_candidates,
)

CONFIG_PATH = os.environ.get("AGENT_CONFIG", "/app/config/config.yaml")
CONFIG = load_config()

AGENT_CFG = CONFIG.get("agent", {})
MODEL = AGENT_CFG.get("model", "qwen3.5:4b")
FAST_MODEL = AGENT_CFG.get("fast_model", "qwen3.5:2b")
MAIN_OPTIONS = AGENT_CFG.get("main_options") or {"num_ctx": 16384, "temperature": 0.4, "top_p": 0.9, "top_k": 20}
FAST_OPTIONS = AGENT_CFG.get("fast_options") or {"num_ctx": 8192, "temperature": 0.0, "top_p": 0.9, "top_k": 20}
MAX_TOOLS_PER_TURN = max(8, int(AGENT_CFG.get("max_tools_per_turn", 12)))
OLLAMA_HOST = AGENT_CFG.get("host", "http://127.0.0.1:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
MAX_CTX = int(AGENT_CFG.get("context", {}).get("num_ctx", MAIN_OPTIONS.get("num_ctx", 16384)))
RESERVE_TOKENS = int(AGENT_CFG.get("context", {}).get("reserve_tokens", 1280))
RECENT_MESSAGES = int(AGENT_CFG.get("context", {}).get("recent_messages", 12))
COMPACT_AT = int(AGENT_CFG.get("context", {}).get("compact_at_tokens", max(8000, int(MAX_CTX * 0.62))))
SUMMARY_KEEP_MESSAGES = int(AGENT_CFG.get("context", {}).get("summary_keep_messages", 8))
MAX_TOOL_OUTPUT = int(AGENT_CFG.get("context", {}).get("max_tool_output_chars", 5000))
TOOL_LOOP_RESERVE = int(AGENT_CFG.get("context", {}).get("tool_loop_reserve_tokens", 4096))
MAX_ITERATIONS = int(AGENT_CFG.get("max_iterations", 12))
SEMANTIC_MEMORY = bool(AGENT_CFG.get("semantic_memory_enabled", False))
THINKING_DEFAULT = bool(AGENT_CFG.get("thinking_default", False))
SHOW_PERF_STATS = bool(AGENT_CFG.get("show_perf_stats", True))
LOOP_VALIDATOR_CFG = AGENT_CFG.get("tool_loop_validator", {})
LOOP_VALIDATOR_ENABLED = bool(LOOP_VALIDATOR_CFG.get("enabled", True))
LOOP_VALIDATOR_OPTIONS = {**FAST_OPTIONS, **(LOOP_VALIDATOR_CFG.get("options") or {})}
LOOP_VALIDATOR_MAX_CHARS = int(LOOP_VALIDATOR_CFG.get("max_transcript_chars", 12000))
LOOP_VALIDATOR_KEEP_ALIVE = LOOP_VALIDATOR_CFG.get("keep_alive", 0)
STALL_VALIDATOR_AFTER = max(2, int(LOOP_VALIDATOR_CFG.get("failed_step_attempts", 3)))
STALL_VALIDATOR_MAX_INTERVENTIONS = max(1, int(LOOP_VALIDATOR_CFG.get("max_interventions_per_turn", 3)))
MAX_TOOL_CALLS_PER_ITERATION = max(1, int(AGENT_CFG.get("max_tool_calls_per_iteration", 3)))
MAX_MUTATING_CALLS_PER_ITERATION = max(1, int(AGENT_CFG.get("max_mutating_calls_per_iteration", 1)))
VISION_CFG = AGENT_CFG.get("vision", {})
AUTO_ATTACH_TOOL_MEDIA = bool(VISION_CFG.get("auto_attach_tool_media", True))
MAX_TOOL_MEDIA_PER_TURN = max(1, int(VISION_CFG.get("max_images_per_turn", 4)))
MAX_MEDIA_BYTES = max(262144, int(VISION_CFG.get("max_image_bytes", 4 * 1024 * 1024)))
SHARED_CTX_CFG = AGENT_CFG.get("model_context_sharing", {})
SHARED_CTX_ENABLED = bool(SHARED_CTX_CFG.get("enabled", True))
SHARED_CTX_MAX_CHARS = max(2000, int(SHARED_CTX_CFG.get("max_chars", 6000)))

OLLAMA = Client(host=OLLAMA_HOST)
LOOP_VALIDATOR_CLIENT = Client(host=OLLAMA_HOST, timeout=float(LOOP_VALIDATOR_CFG.get("timeout_seconds", 45)))

init_db()

BACKGROUND_STATUS = "Idle"
BACKGROUND_STATUS_LOCK = threading.Lock()


def get_bottom_toolbar():
    with BACKGROUND_STATUS_LOCK:
        status = BACKGROUND_STATUS
    return [("class:toolbar", f" ⚙ Background Status: {status} ")]


class Spinner:
    """Short-lived UI spinner. It is never used to schedule work."""
    def __init__(self, msg: str = "Processing"):
        self.msg = str(msg).replace("\n", " ").replace("\r", " ")[:65]
        self.running = False
        self.thread = None

    def _spin(self):
        spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        while self.running:
            sys.stdout.write(f"\r\033[96m{next(spinner)} {self.msg}\033[0m\033[K")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


SYSTEM_POLICY = """
### Agent Runtime Policy
- You are an autonomous local assistant with explicit, typed tools.
- Use deterministic host_snapshot() and network_snapshot() for system awareness instead of repeatedly guessing shell commands.
- For long-running research, use enqueue_research() and get_research_status(). Never use Python threads or sleep to schedule future work.
- For harness improvements, enqueue_self_optimization() may create and test an isolated candidate. Never claim that a candidate is deployed.
- Never approve or promote a self-optimization candidate. Approval requires an explicit human CLI command and promotion requires a separate host command.
- For reminders, use schedule_reminder(), cancel_reminder(), and list_reminders(). Never write systemd unit files yourself.
- Tool calls must contain a valid JSON object matching the tool schema. Never infer missing tool arguments from prose.
- If you need a tool, call it before claiming its result. Do not narrate an expected result as though the tool already succeeded.
- Prefer one necessary tool action at a time. The harness may defer or reject excess calls, especially multiple side-effecting calls.
- Every tool result begins with a trusted harness status line. Treat status=error as a failed attempt that requires a changed argument or approach; do not blindly repeat it.
- After repeated failed/no-progress attempts the harness may inject fast-model recovery guidance. Follow that control guidance and do not repeat an identical failed tool call.
- execute_shell() is a privileged container-local shell tool. Only call it when a structured tool does not provide the required operation and provide an explicit command argument.
- Do not put shell commands, JSON tool-call objects, or tool-call markup in normal assistant prose expecting the harness to execute it.
- Never claim a file write, installation, notification, reminder, job creation, or other side effect succeeded unless a corresponding tool returned harness status=ok in this turn.
- Persist only stable, useful non-sensitive facts with remember(). Avoid secrets, credentials, or ephemeral details.
- Media-producing tools may attach their actual image output to the next model step. Describe visual content only when an image is attached in the current turn; never infer unseen pixels from a filename, URL, or expected website layout.
- If attached media is blank, empty, blocked, or unreadable, say so rather than inventing visual details.
- When a task is long-running, make state recoverable through the durable job/checkpoint tools instead of keeping hidden state only in conversation memory.
"""


def build_system_prompt() -> str:
    """Build a byte-stable system prefix; turn-specific memory is injected later."""
    parts = [AGENT_CFG.get("system_prompt", ""), SYSTEM_POLICY, get_tools_prompt_summary(compact=True)]
    return "\n".join(part for part in parts if part)


def build_memory_context(user_text: str) -> str:
    """Return bounded query-specific memory without changing the system prefix."""
    try:
        memories = get_relevant_memories(user_text, limit=8) if SEMANTIC_MEMORY else json.loads(search_memory(user_text, limit=8))
        return json.dumps(memories, ensure_ascii=False, indent=2)[:5000] if memories else ""
    except Exception:
        return ""


def _clean_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _print_perf_stats(stats: dict[str, Any]) -> None:
    """Display Ollama timing/cache counters when the server returns them."""
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
        fields.append(f"prompt {prompt_count}; cached {cached} ({cached_pct:.1f}%); uncached {uncached}; prefill {prefill_rate:.1f} tok/s")
    if eval_count is not None:
        fields.append(f"generation {eval_count} in {eval_ms:.0f} ms ({generation_rate:.1f} tok/s)")
    if load_ns is not None:
        fields.append(f"load {float(load_ns) / 1_000_000.0:.0f} ms")
    print(f"  \033[90m[Ollama] {'; '.join(fields)}\033[0m")


def append_and_save(messages: list[dict], msg: dict) -> None:
    msg["_db_id"] = _save_message_to_db(msg)
    messages.append(msg)


def encode_image(path_str: str) -> str | None:
    """Encode a local workspace image/PDF or a bounded public image URL."""
    value = str(path_str).strip()
    if value.startswith(("http://", "https://")):
        try:
            from tools.netutil import fetch_bytes
            _, response, body = fetch_bytes(
                value,
                timeout=10,
                max_bytes=MAX_MEDIA_BYTES,
                allowed_types={"image/png", "image/jpeg", "image/webp", "application/pdf"},
            )
            if not response.headers.get("Content-Type", "").split(";", 1)[0].lower() in {"image/png", "image/jpeg", "image/webp", "application/pdf"}:
                return None
            return base64.b64encode(body).decode("ascii")
        except Exception as exc:
            print(f"  \033[93m[System]: Could not download media URL: {exc}\033[0m")
            return None

    candidates = []
    if value.startswith("/app/workspace/"):
        candidates.append(value)
    candidates.extend([
        os.path.join("/app/workspace", value.lstrip("/")),
        os.path.join("/app/workspace", os.path.basename(value)),
    ])
    
    for candidate in candidates:
        safe = os.path.abspath(candidate)
        if os.path.commonpath(["/app/workspace", safe]) != "/app/workspace":
            continue
        if not os.path.isfile(safe):
            continue
        try:
            if safe.lower().endswith(".pdf"):
                from pdf2image import convert_from_path
                pages = convert_from_path(safe, first_page=1, last_page=1)
                if not pages:
                    return None
                buffer = BytesIO()
                pages[0].save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode("ascii")
            with open(safe, "rb") as handle:
                data = handle.read(MAX_MEDIA_BYTES + 1)
            if len(data) > MAX_MEDIA_BYTES:
                return None
            return base64.b64encode(data).decode("ascii")
        except Exception as exc:
            print(f"  \033[93m[System]: Could not encode '{safe}': {exc}\033[0m")
            return None
    return None


IMAGE_REGEX = re.compile(r"(?:https?://[^\s>\"']+\.(?:png|jpg|jpeg|webp|pdf)|/?[\w\-./]+\.(?:png|jpg|jpeg|webp|pdf))", re.I)


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


def _tool_status_prefix(success: bool, reason: str) -> str:
    if success:
        return "[Harness status=ok]"
    safe_reason = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(reason or "tool_error"))[:80]
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


def _bounded_tool_result(tool_name: str, result: Any) -> str:
    """Keep a head/tail preview in context and store the complete observation."""
    text = str(result)
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    observation_id = store_tool_observation(tool_name, text)
    marker = (
        f"\n\n[Harness: middle truncated; full {len(text)}-character result stored as observation "
        f"{observation_id}. Use read_observation(observation_id, offset, length) for another slice.]\n\n"
    )
    remaining = max(200, MAX_TOOL_OUTPUT - len(marker))
    head = remaining // 2
    return text[:head].rstrip() + marker + text[-(remaining - head):].lstrip()


def _finalize_after_limit(messages: list[dict]) -> None:
    """Produce a final answer when the tool loop hits the safety iteration cap."""
    prompt = [
        {"role": "system", "content": build_system_prompt()},
        *build_active_messages(
            system_prompt="",
            summary=get_conversation_summary(),
            history=messages[1:],
            max_ctx_tokens=MAX_CTX,
            reserve_tokens=RESERVE_TOKENS,
            recent_messages=RECENT_MESSAGES,
            extra_prompt_tokens=0,
        )[1:],
        {"role": "user", "content": "The tool-call safety limit was reached. Summarize what has been established, what remains incomplete, and any useful next steps. Do not call tools."},
    ]
    try:
        response = OLLAMA.chat(model=MODEL, messages=prompt, options=MAIN_OPTIONS, tools=[], think=False, keep_alive=-1)
        msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if content:
            print(f"\nAgent: {content}\n")
            append_and_save(messages, {"role": "assistant", "content": content})
    except Exception as exc:
        print(f"  \033[91m[!] Finalization failed: {exc}\033[0m")


def handle_user_turn(messages: list[dict], user_input: str, thinking_enabled: bool) -> None:
    record_monitor_state("agent.last_interaction", utc_now())
    record_monitor_state("agent.interaction_active", {"pid": os.getpid(), "started_at": utc_now()})
    try:
        _prune_compacted_history(messages)
        for previous in messages[1:]:
            previous.pop("images", None)
        msg: dict[str, Any] = {"role": "user", "content": user_input}
        detected_images = []
        for path in IMAGE_REGEX.findall(user_input):
            encoded = encode_image(path)
            if encoded:
                detected_images.append(encoded)
        if detected_images:
            msg["images"] = detected_images
            print(f"  \033[92m[System]: Attached {len(detected_images)} media file(s).\033[0m")
        append_and_save(messages, msg)

        system_prompt = build_system_prompt()
        recent_selection_context = "\n".join(
            str(item.get("content") or "")
            for item in messages[-7:-1]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        )[-3000:]
        tool_schemas = select_tool_schemas(
            user_input,
            max_tools=MAX_TOOLS_PER_TURN,
            context_text=recent_selection_context,
        )
        summary = get_conversation_summary()
        memory_context = build_memory_context(user_input)
        model_user_msg = model_message(msg)
        if memory_context:
            model_user_msg["content"] = (
                "### Relevant stored context\n"
                + memory_context
                + "\n\n### Current request\n"
                + user_input
            )
        model_history = [*messages[1:-1], model_user_msg]
        shared_context = SharedModelContext(
            request=user_input,
            summary=summary if SHARED_CTX_ENABLED else "",
            relevant_memory=memory_context if SHARED_CTX_ENABLED else "",
            recent_messages=messages[-10:-1] if SHARED_CTX_ENABLED else [],
            tool_schemas=tool_schemas if SHARED_CTX_ENABLED else [],
            max_chars=SHARED_CTX_MAX_CHARS,
            summary_chars=int(SHARED_CTX_CFG.get("summary_chars", 2200)),
            recent_chars=int(SHARED_CTX_CFG.get("recent_chars", 1800)),
            memory_chars=int(SHARED_CTX_CFG.get("memory_chars", 1200)),
            tool_chars=int(SHARED_CTX_CFG.get("tool_chars", 1600)),
        )

        def rebuild_prefix() -> tuple[list[dict[str, Any]], int]:
            schema_tokens = estimate_tokens(json.dumps(tool_schemas, ensure_ascii=False, separators=(",", ":")))
            prefix = build_active_messages(
                system_prompt=system_prompt,
                summary=summary,
                history=model_history,
                max_ctx_tokens=MAX_CTX,
                reserve_tokens=RESERVE_TOKENS,
                recent_messages=RECENT_MESSAGES,
                extra_prompt_tokens=schema_tokens + TOOL_LOOP_RESERVE,
            )
            return prefix, schema_tokens

        turn_prefix, tool_prompt_tokens = rebuild_prefix()
        turn_tail: list[dict[str, Any]] = []
        tool_iterations = 0
        seen_tool_calls: set[str] = set()
        successful_mutating_signatures: set[str] = set()
        recovery_validation: dict[str, str] | None = None
        stall_validation: dict[str, str] | None = None
        stall_enforce_once = False
        pending_stall_signal: dict[str, Any] | None = None
        validator_interventions = 0
        tracker = StepFailureTracker(STALL_VALIDATOR_AFTER)

        for iteration in range(1, MAX_ITERATIONS + 1):
            # Mid-loop validator: run before a fourth unvalidated attempt after
            # three deterministic failed/no-progress attempts on one step.
            if pending_stall_signal and LOOP_VALIDATOR_ENABLED:
                if validator_interventions >= STALL_VALIDATOR_MAX_INTERVENTIONS:
                    stall_validation = {"decision": "blocked", "suggested_tool": "", "reason": "intervention limit reached"}
                else:
                    selected_names = {str(schema.get("function", {}).get("name") or "") for schema in tool_schemas}
                    validator_tool_names = sorted(
                        name for name in AVAILABLE_TOOLS_MAP
                        if name in selected_names or bool(TOOL_METADATA.get(name, {}).get("readonly", True))
                    )
                    with Spinner("Fast-model stalled-step validation"):
                        stall_validation = validate_stalled_step(
                            LOOP_VALIDATOR_CLIENT,
                            FAST_MODEL,
                            user_input,
                            turn_tail,
                            pending_stall_signal,
                            validator_tool_names,
                            LOOP_VALIDATOR_OPTIONS,
                            max_chars=LOOP_VALIDATOR_MAX_CHARS,
                            keep_alive=LOOP_VALIDATOR_KEEP_ALIVE,
                            shared_context=shared_context.render() if SHARED_CTX_ENABLED else "",
                        )
                    validator_interventions += 1
                    shared_context.add_validator_event(stall_validation, pending_stall_signal)
                    suggested = str(stall_validation.get("suggested_tool") or "")
                    if suggested:
                        already_selected = suggested in selected_names
                        if not already_selected and _add_recovery_schema(tool_schemas, suggested):
                            shared_context.update_tools(tool_schemas)
                            turn_prefix, tool_prompt_tokens = rebuild_prefix()
                            print(f"  \033[93m[System]: Recovery exposed additional read-only tool schema: {suggested}\033[0m")
                        elif not already_selected:
                            stall_validation["suggested_tool"] = ""
                turn_tail.append({
                    "role": "user",
                    "content": build_stall_recovery_message(stall_validation, pending_stall_signal),
                })
                print(
                    f"  \033[93m[System]: Stalled-step validator decision: "
                    f"{stall_validation['decision']} after {pending_stall_signal.get('attempts', 0)} failed/no-progress attempts.\033[0m"
                )
                pending_stall_signal = None
                stall_enforce_once = True

            # Keep the existing final safety-net validator as a separate last
            # chance if the loop reaches its absolute iteration cap.
            if iteration == MAX_ITERATIONS and tool_iterations and LOOP_VALIDATOR_ENABLED and not stall_enforce_once:
                tool_names = [str(schema.get("function", {}).get("name", "")) for schema in tool_schemas]
                with Spinner("Fast-model tool-loop validation"):
                    recovery_validation = validate_tool_loop(
                        LOOP_VALIDATOR_CLIENT,
                        FAST_MODEL,
                        user_input,
                        turn_tail,
                        [name for name in tool_names if name],
                        LOOP_VALIDATOR_OPTIONS,
                        max_chars=LOOP_VALIDATOR_MAX_CHARS,
                        keep_alive=LOOP_VALIDATOR_KEEP_ALIVE,
                        shared_context=shared_context.render() if SHARED_CTX_ENABLED else "",
                    )
                shared_context.add_validator_event(recovery_validation)
                turn_tail.append({"role": "user", "content": build_recovery_message(recovery_validation)})
                print(f"  \033[93m[System]: Tool-loop validator decision: {recovery_validation['decision']}\033[0m")

            active = fit_tool_loop_messages(
                turn_prefix,
                turn_tail,
                max_ctx_tokens=MAX_CTX,
                reserve_tokens=RESERVE_TOKENS,
                extra_prompt_tokens=tool_prompt_tokens,
            )
            raw_tool_calls = []
            full_content = ""
            in_thinking = False
            in_content = False
            perf_stats: dict[str, Any] = {}
            request_started = time.monotonic()
            first_token_at: float | None = None

            try:
                stream = OLLAMA.chat(
                    model=MODEL,
                    messages=active,
                    tools=tool_schemas,
                    options=MAIN_OPTIONS,
                    think=thinking_enabled,
                    stream=True,
                    keep_alive=-1,
                )
                for chunk in stream:
                    if isinstance(chunk, dict):
                        perf_stats = chunk
                    else:
                        perf_stats = {
                            "done": getattr(chunk, "done", False),
                            "prompt_eval_count": getattr(chunk, "prompt_eval_count", None),
                            "prompt_eval_cached_count": getattr(chunk, "prompt_eval_cached_count", None),
                            "prompt_eval_duration": getattr(chunk, "prompt_eval_duration", None),
                            "eval_count": getattr(chunk, "eval_count", None),
                            "eval_duration": getattr(chunk, "eval_duration", None),
                            "load_duration": getattr(chunk, "load_duration", None),
                        }
                    chunk_msg = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
                    thinking = chunk_msg.get("thinking", "") if isinstance(chunk_msg, dict) else getattr(chunk_msg, "thinking", "")
                    content = chunk_msg.get("content", "") if isinstance(chunk_msg, dict) else getattr(chunk_msg, "content", "")
                    calls = chunk_msg.get("tool_calls", []) if isinstance(chunk_msg, dict) else getattr(chunk_msg, "tool_calls", [])
                    if first_token_at is None and (thinking or content or calls):
                        first_token_at = time.monotonic()
                    if calls:
                        raw_tool_calls = calls
                    if thinking:
                        if not in_thinking:
                            print("\n\033[90m[Thinking Trace]:")
                            in_thinking = True
                        print(thinking, end="", flush=True)
                    if content:
                        if not in_content:
                            print("\033[0m\nAgent: ", end="", flush=True)
                            in_content = True
                        print(content, end="", flush=True)
                        full_content += content
            except Exception as exc:
                print(f"\n\033[91m[!] Ollama error: {exc}\033[0m")
                tracker.record_model_failure("main_inference", str(exc))
                signal = tracker.consume_signal()
                if signal:
                    pending_stall_signal = signal
                if iteration < MAX_ITERATIONS:
                    time.sleep(0.2)
                    continue
                break

            print("\033[0m")
            if first_token_at is not None:
                perf_stats["_ttft_ms"] = (first_token_at - request_started) * 1000.0
            _print_perf_stats(perf_stats)

            supplied_tool_names = {str(schema.get("function", {}).get("name") or "") for schema in tool_schemas}
            parsed_calls, parse_errors = _parse_tool_calls(raw_tool_calls, supplied_tool_names)
            tool_calls, batch_notes = _sanitize_tool_call_batch(parsed_calls, successful_mutating_signatures)
            control_notes = [*parse_errors, *batch_notes]

            if stall_enforce_once and stall_validation is not None:
                emitted_count = len(tool_calls)
                tool_calls = select_stall_recovery_tool_calls(tool_calls, stall_validation, seen_tool_calls)
                if emitted_count > len(tool_calls):
                    control_notes.append("stalled-step recovery suppressed repeated or excess corrective calls")
                stall_enforce_once = False
            elif iteration == MAX_ITERATIONS and recovery_validation is not None:
                emitted_count = len(tool_calls)
                tool_calls = select_recovery_tool_calls(tool_calls, recovery_validation, seen_tool_calls)
                if emitted_count > len(tool_calls):
                    control_notes.append("final recovery suppressed repeated or excess tool calls")

            assistant_msg = {"role": "assistant", "content": full_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            if full_content or tool_calls:
                append_and_save(messages, assistant_msg)
                turn_tail.append(model_message(assistant_msg))

            if not tool_calls:
                recovery_decision = stall_validation.get("decision") if stall_validation else ""
                if recovery_decision in {"finish", "blocked"}:
                    stall_validation = None
                    if not full_content:
                        _finalize_after_limit(messages)
                    break
                if recovery_validation is not None and not full_content:
                    _finalize_after_limit(messages)
                    break

                if raw_tool_calls or control_notes:
                    tracker.record_model_failure("invalid_tool_call", "; ".join(control_notes)[:240])
                    note = "; ".join(control_notes[:4]) or "the emitted call could not be used"
                    turn_tail.append({
                        "role": "user",
                        "content": (
                            "[Harness tool-call correction] The previous tool call was rejected: "
                            f"{note}. Re-read the supplied native tool schemas. Do not invent tool names or arguments. "
                            "Either issue one corrected explicit tool call or answer without tools."
                        ),
                    })
                    signal = tracker.consume_signal()
                    if signal:
                        pending_stall_signal = signal
                    continue

                if not full_content:
                    tracker.record_model_failure("empty_response", "main model emitted neither content nor a valid tool call")
                    turn_tail.append({
                        "role": "user",
                        "content": "[Harness correction] Provide a final answer or issue one explicit valid tool call. Do not emit an empty response.",
                    })
                    signal = tracker.consume_signal()
                    if signal:
                        pending_stall_signal = signal
                    continue

                tracker.clear_model_failure("empty_response")
                break

            tool_iterations += 1
            attached_media: list[str] = []
            attached_from: list[str] = []
            iteration_progress = False
            for call in tool_calls:
                name = call["function"]["name"]
                raw_args = call["function"].get("arguments", {})
                signature = tool_call_signature(call)
                seen_tool_calls.add(signature)
                print(f"\n\033[96m[Tool] {name}\033[0m")

                execution_error = False
                error_reason = ""
                try:
                    args = normalize_arguments(AVAILABLE_TOOLS_MAP[name], raw_args)
                    with Spinner(f"Executing {name}"):
                        result = _execute_registered_tool(name, args)
                except Exception as exc:
                    execution_error = True
                    error_reason = "argument_or_execution_error"
                    result = f"Tool execution error: {exc}"

                result_content, media_refs = unpack_media_result(result)
                encoded_for_tool: list[str] = []
                media_error = False
                if AUTO_ATTACH_TOOL_MEDIA and media_refs and len(attached_media) < MAX_TOOL_MEDIA_PER_TURN:
                    remaining = MAX_TOOL_MEDIA_PER_TURN - len(attached_media)
                    attempted_refs = media_refs[:remaining]
                    for reference in attempted_refs:
                        encoded = encode_image(reference)
                        if encoded:
                            encoded_for_tool.append(encoded)
                            attached_media.append(encoded)
                    if encoded_for_tool:
                        attached_from.append(name)
                        result_content += (
                            f"\n\n[Harness: attached {len(encoded_for_tool)} media item(s) from this tool "
                            "for direct visual inspection on the next model step.]"
                        )
                    elif attempted_refs:
                        media_error = True
                        result_content += (
                            "\n\n[Harness: this tool produced media, but it could not be attached. "
                            "Do not claim to have visually inspected it.]"
                        )
                elif media_refs and len(attached_media) >= MAX_TOOL_MEDIA_PER_TURN:
                    result_content += "\n\n[Harness: additional media was not attached because the per-turn image limit was reached.]"

                outcome = classify_tool_outcome(
                    result_content,
                    tool_name=name,
                    execution_error=execution_error,
                    media_error=media_error,
                )
                success = bool(outcome["success"])
                reason = str(outcome["reason"] if not success else "ok")
                if error_reason and not success:
                    reason = error_reason
                result_with_status = _tool_status_prefix(success, reason) + "\n" + result_content
                result_text = _bounded_tool_result(name, result_with_status)
                print(f"  \033[90m{result_text[:300].replace(chr(10), ' ')}{'...' if len(result_text) > 300 else ''}\033[0m")

                tool_message = {
                    "role": "tool",
                    "name": name,
                    "content": result_text,
                    "tool_call_id": call["id"],
                }
                append_and_save(messages, tool_message)
                turn_tail.append(model_message(tool_message))

                tracker.record_tool(
                    name,
                    success=success,
                    signature=signature,
                    fingerprint=str(outcome.get("fingerprint") or ""),
                    reason=reason,
                )
                if success:
                    iteration_progress = True
                    if not bool(TOOL_METADATA.get(name, {}).get("readonly", True)):
                        successful_mutating_signatures.add(signature)

            if control_notes:
                turn_tail.append({
                    "role": "user",
                    "content": (
                        "[Harness batch note] Some emitted calls were not executed: "
                        + "; ".join(control_notes[:4])
                        + ". Continue only with a distinct necessary action."
                    ),
                })

            if attached_media:
                source_names = ", ".join(dict.fromkeys(attached_from))
                media_message = {
                    "role": "user",
                    "content": (
                        f"[Harness: actual media output from tool(s): {source_names}.] "
                        "Inspect the attached image content directly and use the preceding tool text as context. "
                        "Report what is actually visible. If the image is blank, blocked, or unreadable, state that explicitly. "
                        "Do not infer visual details from the filename, URL, or prior knowledge."
                    ),
                    "images": attached_media,
                }
                turn_tail.append(media_message)
                print(f"  \033[92m[System]: Attached {len(attached_media)} tool media item(s) to the model.\033[0m")

            tracker.record_iteration(made_progress=iteration_progress)
            signal = tracker.consume_signal()
            if signal:
                pending_stall_signal = signal
            stall_validation = None

        else:
            print("\n\033[91m[!] Reached the maximum tool-call iteration limit.\033[0m")
            _finalize_after_limit(messages)

    finally:
        record_monitor_state("agent.interaction_active", False)
        try:
            _queue_compaction_if_needed(messages)
        except Exception as exc:
            print(f"  \033[93m[System]: Could not queue context compaction: {exc}\033[0m")


def print_jobs() -> None:
    jobs = list_jobs(limit=25)
    if not jobs:
        print("\n[System]: No durable background jobs.")
        return
    print("\n[System]: Durable jobs:")
    for job in jobs:
        print(f"  {job['id'][:8]}  {job['status']:<10} {job['job_type']:<10} {job['title']}")


def print_job(job_id: str) -> None:
    status = get_research_status(job_id)
    print("\n" + status)


def main() -> None:
    global BACKGROUND_STATUS
    SHELL_HISTORY_FILE = "/app/memory/.agent_history"
    os.makedirs(os.path.dirname(SHELL_HISTORY_FILE), exist_ok=True)
    style = Style.from_dict({
        "prompt": "ansigreen bold",
        "input": "ansiwhite",
        "toolbar": "bg:#1a202c #63b3ed bold",
        "completion-menu.completion": "bg:#000000 #ffffff",
        "completion-menu.completion.current": "bg:#005f5f #ffffff",
    })
    session = PromptSession(
        history=FileHistory(SHELL_HISTORY_FILE),
        auto_suggest=AUTO_SUGGEST,
        style=style,
        bottom_toolbar=get_bottom_toolbar,
    )
    raw = _load_chat_history_from_db(limit=100)
    messages = [{"role": "system", "content": build_system_prompt()}] + raw
    thinking_enabled = THINKING_DEFAULT

    print(f"Agent initialized with Main: {MODEL} | Fast: {FAST_MODEL} | Context: {MAX_CTX}")
    
    print(f"[System]: Preloading main model ({MODEL}) into VRAM...")
    try:
        OLLAMA.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "warmup"}],
            options=MAIN_OPTIONS,
            keep_alive=-1,
            think=False,
        )
        print("[System]: Main model successfully loaded and pinned in VRAM.")
    except Exception as exc:
        print(f"[System]: Warning - failed to preload main model: {exc}")

    print("Commands: /research, /optimize, /optimizations, /approve-optimization, /jobs, /job, /cancel-job, /reminders, /tools, /think, /reload, /forget, exit")
    print("Background research runs in the durable worker process and survives CLI restarts.")

    while True:
        try:
            with patch_stdout():
                user_input = session.prompt("\nYou: ").strip()
            if not user_input:
                continue
            lowered = user_input.lower()
            if lowered in {"exit", "quit"}:
                break

            if lowered.startswith("/research"):
                topic = user_input[len("/research"):].strip()
                if not topic:
                    print("[System]: Usage: /research <topic>")
                    continue
                result = json.loads(enqueue_research(topic))
                print(f"[System]: queued research job {result['job_id'][:8]} for '{topic}'")
                continue

            if lowered.startswith("/optimize"):
                objective = user_input[len("/optimize"):].strip()
                if not objective:
                    print("[System]: Usage: /optimize <bounded objective>")
                    continue
                result = json.loads(enqueue_self_optimization(objective))
                if result.get("job_id"):
                    print(f"[System]: queued isolated candidate {result['candidate_id'][:8]} as job {result['job_id'][:8]}")
                else:
                    print(f"[System]: {result.get('reason', 'Optimization was not queued.')}")
                continue
            if lowered == "/optimizations":
                print("\n" + list_self_optimization_candidates())
                continue
            if lowered.startswith("/approve-optimization "):
                parts = user_input.split()
                if len(parts) != 3:
                    print("[System]: Usage: /approve-optimization <candidate-id> <full-sha256>")
                    continue
                print("\n" + approve_self_optimization(parts[1], parts[2]))
                continue

            if lowered == "/jobs":
                print_jobs()
                continue
            if lowered.startswith("/job "):
                print_job(user_input.split(None, 1)[1].strip())
                continue
            if lowered.startswith("/cancel-job "):
                from tools.job_tools import cancel_background_job
                print(cancel_background_job(user_input.split(None, 1)[1].strip()))
                continue
            if lowered == "/reminders":
                from tools.reminders import list_reminders
                print("\n" + list_reminders())
                continue
            if lowered == "/forget":
                clear_chat_history()
                messages = [{"role": "system", "content": build_system_prompt()}]
                print("[System]: Conversation history and rolling summary cleared.")
                continue
            if lowered.startswith("/think"):
                parts = lowered.split()
                thinking_enabled = (parts[1] in {"on", "true", "1"}) if len(parts) > 1 else not thinking_enabled
                print(f"[System]: Thinking is {'ON' if thinking_enabled else 'OFF'}.")
                continue
            if lowered == "/tools":
                print(get_tools_prompt_summary())
                continue
            if lowered == "/reload":
                count, errors = load_tools()
                print(f"[System]: Reloaded {count} tools.")
                if errors:
                    for key, value in errors.items():
                        print(f"  - {key}: {value}")
                continue

            handle_user_turn(messages, user_input, thinking_enabled)

        except KeyboardInterrupt:
            print("\n[System]: Use 'exit' or 'quit' to close.")
        except EOFError:
            break
        except Exception as exc:
            print(f"\n[System] Unhandled frontend error: {exc}")


if __name__ == "__main__":
    main()
