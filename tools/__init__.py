"""Stable public facade for tools, memory, and registry APIs.

Implementation details live in focused modules (``catalog``, ``providers``,
``tool_registry`` and domain tool modules).  Importing from ``tools`` remains
backward compatible for the worker, Web UI, tests, and custom extensions.
"""
from .catalog import (
    ALL_TOOLS,
    AVAILABLE_TOOLS_MAP,
    TOOL_METADATA,
    TOOL_SCHEMAS,
    get_tool_schema,
    get_tools_prompt_summary,
    load_tools,
    select_tool_schemas,
)
from .tool_registry import agent_tool, adapt_tool_schemas_for_qwen, normalize_arguments
from .memory import (
    _init_chat_db,
    _init_checkpoint_db,
    _load_chat_history_from_db,
    _save_message_to_db,
    apply_conversation_compaction,
    clear_chat_history,
    create_conversation,
    delete_conversation,
    ensure_conversation,
    list_conversations,
    rename_conversation,
    get_all_memories_prompt_summary,
    get_compacted_through_id,
    get_conversation_summary,
    get_messages_for_compaction,
    get_relevant_memories,
    init_db,
    read_observation,
    search_memory,
    search_conversation_history,
    set_conversation_summary,
    store_tool_observation,
)

__all__ = [
    "ALL_TOOLS", "AVAILABLE_TOOLS_MAP", "TOOL_SCHEMAS", "TOOL_METADATA",
    "load_tools", "get_tools_prompt_summary", "select_tool_schemas", "get_tool_schema", "adapt_tool_schemas_for_qwen", "normalize_arguments",
    "init_db", "_init_chat_db", "_init_checkpoint_db", "_load_chat_history_from_db",
    "_save_message_to_db", "clear_chat_history", "create_conversation", "delete_conversation", "ensure_conversation", "list_conversations", "rename_conversation", "get_all_memories_prompt_summary",
    "get_conversation_summary", "set_conversation_summary", "get_compacted_through_id",
    "get_messages_for_compaction", "apply_conversation_compaction", "store_tool_observation",
    "read_observation", "get_relevant_memories", "search_memory", "search_conversation_history", "agent_tool",
]

# Backward-compatible selector-policy aliases used by tests/extensions.
from .providers import ALWAYS_TOOL_NAMES as _ALWAYS_TOOL_NAMES

from .conversation_context import DEFAULT_CONVERSATION_ID, conversation_context, get_active_conversation_id, normalize_conversation_id
__all__ += ["DEFAULT_CONVERSATION_ID", "conversation_context", "get_active_conversation_id", "normalize_conversation_id"]
