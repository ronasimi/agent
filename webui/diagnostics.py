"""On-demand diagnostic report capture for the Web UI.

The report is intentionally source-oriented: it records the runtime state and
recent wire-level model interactions that are difficult to reconstruct from a
source archive alone. Capture is explicit, bounded, local, and secret-aware.
"""
from __future__ import annotations

import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from al_agent import runtime as agent_runtime
from tools.memory import (
    _load_chat_history_from_db,
    ensure_conversation,
    get_compacted_through_id,
    get_conversation_summary,
)
from tools.runtime import DB_PATH, list_jobs, list_monitor_events
from tools.working_state import WorkingStateStore

_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:access_token|refresh_token|id_token|client_secret|password|passwd|authorization|"
    r"api_key|apikey|credential|cookie|session_cookie|master_key|private_key)(?:$|_)",
    re.I,
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:access_token|refresh_token|client_secret|password|api[_-]?key)\b\s*[:=]\s*)"
    r"([^\s,}\]]+|\"[^\"]*\")"
)
_MAX_SECTION_CHARS = 120_000
_MAX_VALUE_CHARS = 24_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _redact_text(value: str) -> str:
    text = _BEARER_RE.sub("Bearer [REDACTED]", str(value))
    return _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", text)


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        return "[MAX_DEPTH]"
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _SECRET_KEY_RE.search(name):
                clean[name] = "[REDACTED]"
            else:
                clean[name] = _sanitize(item, depth=depth + 1)
        return clean
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        text = _redact_text(value)
        if len(text) > _MAX_VALUE_CHARS:
            return text[:_MAX_VALUE_CHARS] + f"\n...[truncated {len(text) - _MAX_VALUE_CHARS} chars]"
        return text
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


def _json_block(value: Any, *, max_chars: int = _MAX_SECTION_CHARS) -> str:
    text = json.dumps(_sanitize(value), ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[section truncated {len(text) - max_chars} chars]"
    return text


def _run_git(source_root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=source_root, text=True, capture_output=True,
            timeout=2.5, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    output = (proc.stdout or proc.stderr or "").strip()
    return output[:20_000] if output else f"exit={proc.returncode}"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("ollama", "fastapi", "uvicorn", "pydantic", "PyYAML", "httpx", "websockets"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _db_diagnostics() -> dict[str, Any]:
    path = Path(str(DB_PATH))
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        result["size_bytes"] = path.stat().st_size
    except OSError:
        pass
    tables = (
        "chat_history", "conversations", "conversation_context", "memory",
        "working_states", "tool_observations", "agent_jobs", "monitor_events",
        "reminders", "browser_benchmark_runs",
    )
    try:
        with sqlite3.connect(str(path), timeout=2.0) as conn:
            result["journal_mode"] = conn.execute("PRAGMA journal_mode").fetchone()[0]
            result["synchronous"] = conn.execute("PRAGMA synchronous").fetchone()[0]
            existing = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
            }
            counts: dict[str, int] = {}
            for table in tables:
                if table in existing:
                    counts[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            result["row_counts"] = counts
            result["fts5"] = {
                "chat_history_fts": "chat_history_fts" in existing,
                "memory_fts": "memory_fts" in existing,
            }
    except (sqlite3.Error, OSError) as exc:
        result["error"] = str(exc)
    return result


def _recent_model_traces(conversation_id: str, *, limit: int = 12, scan_lines: int = 400) -> list[dict[str, Any]]:
    path = Path(str(agent_runtime.MODEL_TRACE_PATH))
    if not path.is_file():
        return []
    lines: deque[str] = deque(maxlen=max(limit, scan_lines))
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lines.append(line)
    except OSError:
        return []
    selected: list[dict[str, Any]] = []
    for raw in reversed(lines):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if conversation_id and str(row.get("conversation_id") or "") != conversation_id:
            continue
        selected.append(_sanitize(row))
        if len(selected) >= limit:
            break
    selected.reverse()
    return selected


def _runtime_snapshot(conversation_id: str) -> dict[str, Any]:
    try:
        from al_agent.model_capabilities import get_active_model_capabilities
        capability_profiles = {}
        for role, model in (("main", agent_runtime.MODEL), ("fast", agent_runtime.FAST_MODEL), ("vision", agent_runtime.VISION_MODEL)):
            profile = get_active_model_capabilities(model)
            if profile is not None:
                capability_profiles[role] = profile.to_dict()
    except Exception:
        capability_profiles = {}
    return {
        "conversation_id": conversation_id,
        "models": {
            "main": agent_runtime.MODEL,
            "fast": agent_runtime.FAST_MODEL,
            "vision": agent_runtime.VISION_MODEL,
            "report": str(agent_runtime.AGENT_CFG.get("report_model") or ""),
        },
        "context": {
            "main_num_ctx": agent_runtime.MAIN_OPTIONS.get("num_ctx"),
            "fast_num_ctx": agent_runtime.FAST_OPTIONS.get("num_ctx"),
            "max_context": agent_runtime.MAX_CTX,
            "working_state_enabled": agent_runtime.WORKING_STATE_ENABLED,
            "working_state_history_turns": agent_runtime.WORKING_STATE_HISTORY_TURNS,
            "semantic_memory_enabled": agent_runtime.SEMANTIC_MEMORY,
            "configured_context": agent_runtime.AGENT_CFG.get("context", {}),
        },
        "generation": {
            "thinking_default": agent_runtime.THINKING_DEFAULT,
            "tool_turn_num_predict": agent_runtime.TOOL_TURN_NUM_PREDICT,
            "final_num_predict": agent_runtime.FINAL_NUM_PREDICT,
            "model_transport_timeout": agent_runtime.MODEL_TRANSPORT_TIMEOUT,
            "max_model_calls_per_turn": agent_runtime.MAX_MODEL_CALLS_PER_TURN,
            "structured_plan_max_model_calls": agent_runtime.STRUCTURED_PLAN_MAX_MODEL_CALLS,
            "structured_plan_max_iterations": agent_runtime.STRUCTURED_PLAN_MAX_ITERATIONS,
            "structured_plan_soft_timeout_seconds": agent_runtime.STRUCTURED_PLAN_SOFT_TIMEOUT_SECONDS,
            "structured_plan_hard_timeout_seconds": agent_runtime.STRUCTURED_PLAN_HARD_TIMEOUT_SECONDS,
            "main_options": agent_runtime.MAIN_OPTIONS,
            "fast_options": agent_runtime.FAST_OPTIONS,
        },
        "features": {
            "grounding": agent_runtime.GROUNDING_ENABLED,
            "recipes": agent_runtime.RECIPES_ENABLED,
            "model_traces": agent_runtime.MODEL_TRACE_ENABLED,
            "shared_model_context": agent_runtime.SHARED_CTX_ENABLED,
        },
        "model_capabilities": capability_profiles,
        "ollama_host": str(agent_runtime.OLLAMA_HOST),
        "model_trace_path": str(agent_runtime.MODEL_TRACE_PATH),
    }



def _repository_inventory(source_root: Path) -> dict[str, Any]:
    """Return a compact source map without embedding the codebase itself."""
    ignored = {".git", ".venv", "venv", "workspace", "memory", "__pycache__", ".pytest_cache", ".ruff_cache"}
    roots: dict[str, Any] = {"root_files": [], "directories": {}}
    try:
        for child in sorted(source_root.iterdir(), key=lambda item: item.name.lower()):
            if child.name in ignored or child.name.startswith("al-agent-bug-report-"):
                continue
            if child.is_file():
                roots["root_files"].append(child.name)
                continue
            if not child.is_dir():
                continue
            files: list[str] = []
            for path in sorted(child.rglob("*")):
                if not path.is_file() or any(part in ignored for part in path.relative_to(source_root).parts):
                    continue
                rel = path.relative_to(source_root).as_posix()
                files.append(rel)
                if len(files) >= 80:
                    break
            roots["directories"][child.name] = {
                "sample_files": files,
                "sample_truncated": len(files) >= 80,
            }
    except OSError as exc:
        roots["error"] = str(exc)
    return roots


def _failure_signals(
    working_state: dict[str, Any], traces: list[dict[str, Any]], monitor_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect high-signal failure facts near the top of a report."""
    recent_trace_failures: list[dict[str, Any]] = []
    for row in traces:
        completion = row.get("completion") if isinstance(row, dict) else None
        metrics = row.get("metrics") if isinstance(row, dict) else None
        error = row.get("error") if isinstance(row, dict) else None
        done_reason = metrics.get("done_reason") if isinstance(metrics, dict) else None
        if error or done_reason not in (None, "", "stop"):
            recent_trace_failures.append({
                "at": row.get("at"),
                "turn_id": row.get("turn_id"),
                "purpose": row.get("purpose"),
                "error": error,
                "done_reason": done_reason,
                "completion": completion,
            })
    return {
        "working_state_status": working_state.get("status"),
        "objective": working_state.get("objective"),
        "failed_approaches": working_state.get("failed_approaches", []),
        "open_questions": working_state.get("open_questions", []),
        "validator_history": working_state.get("validator_history", []),
        "recent_model_failures": recent_trace_failures[-8:],
        "recent_monitor_events": monitor_events[-12:],
    }


def generate_bug_report(source_root: Path, conversation_id: str = "default") -> dict[str, Any]:
    """Generate a bounded, redacted LLM troubleshooting report in the repo root."""
    generated = _utc_now()
    cid = ensure_conversation(conversation_id)
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"Repository root does not exist: {source_root}")
    stamp = generated.strftime("%Y%m%d-%H%M%SZ")
    filename = f"al-agent-bug-report-{stamp}.md"
    target = (source_root / filename).resolve()
    if target.parent != source_root:
        raise RuntimeError("Bug-report target escaped the repository root")

    history = _load_chat_history_from_db(limit=50, include_compacted=True, conversation_id=cid)
    summary = get_conversation_summary(cid)
    compacted_through = get_compacted_through_id(cid)
    working_state = WorkingStateStore(
        limits=agent_runtime.WORKING_STATE_CFG, conversation_id=cid,
    ).load()
    traces = _recent_model_traces(cid, limit=16, scan_lines=600)
    monitor_events = list_monitor_events(limit=40)
    jobs = list_jobs(limit=30)

    repository = {
        "source_root": str(source_root),
        "git_commit": _run_git(source_root, "rev-parse", "HEAD"),
        "git_branch": _run_git(source_root, "branch", "--show-current"),
        "git_status": _run_git(source_root, "status", "--short"),
        "git_recent_commits": _run_git(source_root, "log", "--oneline", "-8"),
        "git_diff_stat": _run_git(source_root, "diff", "--stat"),
        "git_diff": _run_git(source_root, "diff", "--no-ext-diff", "--unified=3"),
        "source_map": _repository_inventory(source_root),
    }
    process = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "packages": _package_versions(),
        "selected_environment": {
            name: os.environ.get(name, "")
            for name in (
                "AGENT_WORKSPACE", "AGENT_DB_PATH", "AGENT_SOURCE_ROOT", "OLLAMA_HOST",
                "WEBUI_HOST", "WEBUI_PORT", "AGENT_INFERENCE_LOCK", "TZ",
            )
        },
    }

    try:
        from tools.repo_diagnostics import tool_health
        tool_health_payload = json.loads(tool_health())
    except Exception as exc:
        tool_health_payload = {"error": str(exc)}
    try:
        from tools.browser_benchmark import browsergym_available, regression_dashboard
        benchmark_payload = regression_dashboard(limit=30)
        benchmark_payload["browsergym"] = browsergym_available()
    except Exception as exc:
        benchmark_payload = {"error": str(exc)}

    sections: list[tuple[str, Any]] = [
        ("Triage summary", _failure_signals(working_state, traces, monitor_events)),
        ("Runtime snapshot", _runtime_snapshot(cid)),
        ("Effective agent configuration", agent_runtime.AGENT_CFG),
        ("Effective base system prompt", agent_runtime.build_system_prompt()),
        ("Active working state", working_state),
        ("Rolling conversation summary", {
            "compacted_through_message_id": compacted_through,
            "summary": summary,
        }),
        ("Recent timestamped conversation history", history),
        ("Recent model-call wire traces", traces),
        ("Durable jobs", jobs),
        ("Recent monitor events", monitor_events),
        ("Tool registry/dependency health", tool_health_payload),
        ("Browser/UI benchmark state", benchmark_payload),
        ("Database/storage health", _db_diagnostics()),
        ("Repository state and source map", repository),
        ("Process/runtime versions", process),
    ]

    lines = [
        "# Al Agent Bug Report",
        "",
        f"Generated (UTC): `{generated.isoformat(timespec='seconds')}`",
        f"Conversation: `{cid}`",
        f"Repository root: `{source_root}`",
        "",
        "> Generated on demand for LLM-assisted troubleshooting alongside the matching codebase.",
        "> Known credential/token fields are redacted. Conversation text and effective system prompts may be included because they are often necessary to reproduce routing, context, and model-protocol failures.",
        "",
        "## Suggested diagnostic use",
        "",
        "Provide this report together with the repository revision/archive that produced it. Start with the Triage summary, recent model traces, working state, and git diff before inspecting the referenced source modules.",
        "",
    ]
    for title, payload in sections:
        lines.extend([f"## {title}", "", "```json" if not isinstance(payload, str) else "```text"])
        if isinstance(payload, str):
            value = _redact_text(payload)
            if len(value) > _MAX_SECTION_CHARS:
                value = value[:_MAX_SECTION_CHARS] + "\n...[section truncated]"
            lines.append(value)
        else:
            lines.append(_json_block(payload))
        lines.extend(["```", ""])

    try:
        target.write_text("\n".join(lines), encoding="utf-8")
    except PermissionError as exc:
        raise RuntimeError(
            f"Repository root is not writable from the Web UI container: {source_root}. "
            "Ensure the /app/source bind is writable for the webui service."
        ) from exc
    return {
        "ok": True,
        "filename": filename,
        "path": str(target),
        "relative_path": filename,
        "created_at": generated.isoformat(timespec="seconds"),
        "conversation_id": cid,
        "size_bytes": target.stat().st_size,
    }


# Backward-compatible Python alias for integrations that imported the old helper.
def capture_diagnostic_report(source_root: Path, conversation_id: str = "default") -> dict[str, Any]:
    return generate_bug_report(source_root, conversation_id)
