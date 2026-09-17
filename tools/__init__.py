"""Builtin and custom agent tool registry.

The language model receives explicit JSON schemas.  Tool execution is resolved
through this allowlist rather than by exposing every public function imported
from every Python module.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path

from .tool_registry import agent_tool, function_schema, normalize_arguments

ALL_TOOLS = []
AVAILABLE_TOOLS_MAP = {}
TOOL_SCHEMAS = []
TOOL_METADATA = {}

# Intentional allowlist. Powerful shell/code tools are still available but are
# explicit calls, never inferred from free-form model output.
MUTATING_TOOLS = {
    "remember", "remember_semantic",
    "enqueue_research", "cancel_background_job",
    "schedule_reminder", "cancel_reminder",
    "notify_desktop", "write_file", "generate_pdf_report", "take_web_screenshot",
    "install_package", "execute_shell", "execute_python",
    "queue_work", "update_work_status", "map_network",
    "create_or_update_tool", "reload_tools",
}
BUILTINS = [
    ("memory", "remember"), ("memory", "search_memory"),
    ("memory", "remember_semantic"), ("memory", "search_semantic_memory"),
    ("job_tools", "enqueue_research"), ("job_tools", "get_research_status"),
    ("job_tools", "list_background_jobs"), ("job_tools", "cancel_background_job"),
    ("reminders", "schedule_reminder"), ("reminders", "cancel_reminder"), ("reminders", "list_reminders"),
    ("notify", "notify_desktop"),
    ("host_tools", "host_snapshot"), ("host_tools", "gpu_snapshot_dict"),
    ("host_tools", "ollama_runtime_snapshot"), ("host_tools", "network_snapshot"),
    ("host_tools", "network_reachability"), ("host_tools", "list_host_monitor_events"), ("host_tools", "read_host_file"),
    ("host_tools", "read_host_journal"), ("host_tools", "tail_host_log"),
    ("web", "web_search"), ("web", "wiki_search"), ("web", "browse_url"),
    ("web_screenshot", "take_web_screenshot"),
    ("pdf_generator", "generate_pdf_report"),
    ("workspace", "read_file"), ("workspace", "write_file"),
    ("packages", "search_packages"), ("packages", "install_package"),
    ("system", "execute_shell"), ("system", "execute_python"),
    ("network_mapper", "map_network"), ("mdns_scanner", "scan_mdns"),
    ("work_queue", "queue_work"), ("work_queue", "list_work_queue"),
    ("work_queue", "get_work_details"), ("work_queue", "get_work_result"),
    ("work_queue", "update_work_status"), ("work_queue", "get_work_statistics"),
    ("task_manager", "list_tasks"), ("task_manager", "get_task_info"), ("task_manager", "get_task_logs"),
    ("tool_manager", "list_tool_files"), ("tool_manager", "create_or_update_tool"),
    ("tool_manager", "read_tool_source"), ("tool_manager", "reload_tools"),
]


def _register(func, *, builtin_name: str | None = None) -> None:
    public_name = getattr(func, "_agent_tool_name", None) or builtin_name or func.__name__
    
    # Evaluate schema first so a failure doesn't leave the registry in a partially mutated state
    schema = function_schema(func)
    
    AVAILABLE_TOOLS_MAP[public_name] = func
    if func not in ALL_TOOLS:
        ALL_TOOLS.append(func)
    TOOL_SCHEMAS.append(schema)
    TOOL_METADATA[public_name] = {
        "readonly": False if public_name in MUTATING_TOOLS else bool(getattr(func, "_agent_tool_readonly", True)),
        "timeout": getattr(func, "_agent_tool_timeout", None),
        "function": func,
    }


def load_tools() -> tuple[int, dict[str, str]]:
    """Load the explicit builtin allowlist and decorated workspace custom tools."""
    ALL_TOOLS.clear(); AVAILABLE_TOOLS_MAP.clear(); TOOL_SCHEMAS.clear(); TOOL_METADATA.clear()
    errors: dict[str, str] = {}

    for module_name, function_name in BUILTINS:
        try:
            module = __import__(f"tools.{module_name}", fromlist=[function_name])
            func = getattr(module, function_name)
            _register(func, builtin_name=function_name)
        except Exception as exc:
            errors[f"{module_name}.{function_name}"] = str(exc)

    custom_dir = Path("/app/workspace/custom_tools")
    custom_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(custom_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module_name = f"agent_custom_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Could not load module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for _, func in inspect.getmembers(module, inspect.isfunction):
                if getattr(func, "_agent_tool", False):
                    _register(func)
        except Exception as exc:
            errors[f"custom:{path.name}"] = str(exc)
    return len(AVAILABLE_TOOLS_MAP), errors


def get_tools_prompt_summary() -> str:
    """Return a compact text inventory; the actual schemas are sent natively to Ollama."""
    lines = ["\n\n### Tool policy", "Tools are explicitly typed. Never infer a missing argument from natural-language output."]
    for name, func in AVAILABLE_TOOLS_MAP.items():
        doc = (inspect.getdoc(func) or "No description.").splitlines()[0]
        flags = []
        if TOOL_METADATA[name].get("readonly"):
            flags.append("read-only")
        else:
            flags.append("mutating")
        lines.append(f"- **{name}** ({', '.join(flags)}): {doc}")
    return "\n".join(lines)


load_tools()

from .memory import (
    init_db,
    _init_chat_db,
    _init_checkpoint_db,
    _load_chat_history_from_db,
    _save_message_to_db,
    clear_chat_history,
    get_all_memories_prompt_summary,
    get_conversation_summary,
    set_conversation_summary,
    get_relevant_memories,
    search_memory,
)

__all__ = [
    "ALL_TOOLS", "AVAILABLE_TOOLS_MAP", "TOOL_SCHEMAS", "TOOL_METADATA",
    "load_tools", "get_tools_prompt_summary", "normalize_arguments",
    "init_db", "_init_chat_db", "_init_checkpoint_db", "_load_chat_history_from_db",
    "_save_message_to_db", "clear_chat_history", "get_all_memories_prompt_summary",
    "get_conversation_summary", "set_conversation_summary", "get_relevant_memories",
    "search_memory", "agent_tool",
]
