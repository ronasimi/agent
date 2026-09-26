"""Single-model configuration and long-lived services."""

from __future__ import annotations

import os

from ollama import Client

from tools.config import config_path, load_config
from tools.memory import init_db
from tools.recipe_compat import seed_builtin_recipes
from tools.recipe_store import init_recipe_store
from tools.working_state import WorkingStateStore

from .deterministic_router import DeterministicToolRouter

CONFIG_PATH = str(config_path())
CONFIG = load_config()
AGENT_CFG = CONFIG["agent"]
MODEL = AGENT_CFG["model"]
MAIN_OPTIONS = dict(AGENT_CFG["main_options"])
TOOL_ROUTING_CFG = dict(AGENT_CFG.get("tool_routing") or {})
TOOL_ROUTING_CANDIDATES = int(TOOL_ROUTING_CFG.get("candidate_limit", 8))
TOOL_ROUTING_AUTO_THRESHOLD = float(
    TOOL_ROUTING_CFG.get("auto_activate_threshold", 0.80)
)
TOOL_ROUTING_AUTO_MARGIN = float(TOOL_ROUTING_CFG.get("auto_activate_margin", 0.20))
TOOL_ROUTING_MIN_SCORE = float(TOOL_ROUTING_CFG.get("min_candidate_score", 0.18))
OLLAMA_HOST = AGENT_CFG["host"]
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
# Compatibility names do not define roles or alternate runners.
EXECUTOR_MODEL = DECISION_MODEL = REASONING_MODEL = FAST_MODEL = VISION_MODEL = MODEL
FAST_OPTIONS = DECISION_OPTIONS = REASONING_OPTIONS = VISION_OPTIONS = MAIN_OPTIONS
FAST_MODEL_KEEP_ALIVE = DECISION_MODEL_KEEP_ALIVE = REASONING_MODEL_KEEP_ALIVE = (
    VISION_MODEL_KEEP_ALIVE
) = AGENT_CFG.get("keep_alive", -1)
MAX_CTX = int(MAIN_OPTIONS["num_ctx"])
CONTEXT_CFG = AGENT_CFG.get("context", {})
RESERVE_TOKENS = max(
    int(MAIN_OPTIONS.get("num_predict", 2048)),
    int(CONTEXT_CFG.get("reserve_tokens", 2560)),
)
RECENT_MESSAGES = int(CONTEXT_CFG.get("recent_messages", 100))
RECENT_CONVERSATION_TURNS = max(1, int(CONTEXT_CFG.get("recent_conversation_turns", 4)))
RECENT_CONVERSATION_USER_CHARS = max(400, int(CONTEXT_CFG.get("recent_conversation_user_chars", 1400)))
RECENT_CONVERSATION_ASSISTANT_CHARS = max(400, int(CONTEXT_CFG.get("recent_conversation_assistant_chars", 900)))
STATE_TAPE_RECENT_ENTRIES = max(1, int(CONTEXT_CFG.get("state_tape_entries", 6)))
STATE_TAPE_UNRESOLVED_ENTRIES = max(1, int(CONTEXT_CFG.get("state_tape_unresolved_entries", 3)))
STATE_TAPE_ENTRY_CHARS = max(180, int(CONTEXT_CFG.get("state_tape_entry_chars", 440)))
STATE_TAPE_ROLLING_SUMMARY_CHARS = max(800, int(CONTEXT_CFG.get("rolling_summary_chars", 2400)))
HARD_PROMPT_TOKENS = min(
    max(128, int(CONTEXT_CFG.get("hard_prompt_tokens", MAX_CTX - RESERVE_TOKENS))),
    max(128, MAX_CTX - RESERVE_TOKENS),
)
SOFT_PROMPT_TOKENS = min(
    max(128, int(CONTEXT_CFG.get("soft_prompt_tokens", 8192))),
    HARD_PROMPT_TOKENS,
)
MAX_TOOL_OUTPUT = int(CONTEXT_CFG.get("max_tool_output_chars", 6000))
MAX_MODEL_CALLS_PER_TURN = int(AGENT_CFG.get("max_model_calls_per_turn", 24))
MAX_TOOL_CALLS_PER_ITERATION = int(AGENT_CFG.get("max_tool_calls_per_iteration", 6))
MODEL_NO_PROGRESS_MAX_RETRIES = int(AGENT_CFG.get("model_no_progress_max_retries", 3))
TURN_HARD_TIMEOUT_SECONDS = float(AGENT_CFG.get("turn_hard_timeout_seconds", 600))
MODEL_TRANSPORT_CFG = AGENT_CFG.get("model_transport", {})
MODEL_FIRST_BYTE_TIMEOUT = float(
    MODEL_TRANSPORT_CFG.get("first_byte_timeout_seconds", MODEL_TRANSPORT_CFG.get("timeout_seconds", 120))
)
MODEL_STREAM_IDLE_TIMEOUT = float(
    MODEL_TRANSPORT_CFG.get("stream_idle_timeout_seconds", 60)
)
# Compatibility alias used by background helpers; foreground streaming has a
# separate first-byte and post-first-chunk idle deadline.
MODEL_TRANSPORT_TIMEOUT = MODEL_FIRST_BYTE_TIMEOUT
INFERENCE_LOCK_TIMEOUT_SECONDS = float(
    MODEL_TRANSPORT_CFG.get("queue_timeout_seconds", 90)
)
INFERENCE_LOCK_PATH = os.environ.get(
    "AGENT_INFERENCE_LOCK", "/app/workspace/.agent_inference.lock"
)
THINKING_DEFAULT = bool(AGENT_CFG.get("thinking_default", False))
SEMANTIC_MEMORY = False
SHOW_PERF_STATS = bool(AGENT_CFG.get("show_perf_stats", True))
VISION_CFG = AGENT_CFG.get("vision", {})
VISION_SUPPORTS_IMAGES = bool(VISION_CFG.get("supports_images", False))
MAX_MEDIA_BYTES = int(VISION_CFG.get("max_image_bytes", 4194304))
WARMUP_CFG = AGENT_CFG.get("warmup", {})
WARMUP_ENABLED = bool(WARMUP_CFG.get("enabled", True))
WARMUP_PRIME_PREFIX = bool(WARMUP_CFG.get("prime_system_prefix", False))
WORKING_STATE_CFG = AGENT_CFG.get("working_state", {})
WORKING_STATE_ENABLED = bool(WORKING_STATE_CFG.get("enabled", True))
WORKING_STATE_HISTORY_TURNS = 1
RECIPE_CFG = AGENT_CFG.get("recipes", {})
RECIPES_ENABLED = bool(RECIPE_CFG.get("enabled", True))
MODEL_TRACE_CFG = AGENT_CFG.get("model_traces", {})
MODEL_TRACE_ENABLED = bool(MODEL_TRACE_CFG.get("enabled", True))
MODEL_TRACE_PATH = str(MODEL_TRACE_CFG.get("path", "/app/memory/model_calls.jsonl"))
MODEL_TRACE_MAX_BYTES = int(MODEL_TRACE_CFG.get("max_bytes", 268435456))
OLLAMA = Client(host=OLLAMA_HOST, timeout=MODEL_TRANSPORT_TIMEOUT)
init_db()
TOOL_ROUTER = DeterministicToolRouter(
    candidate_count=TOOL_ROUTING_CANDIDATES,
    auto_activate_threshold=TOOL_ROUTING_AUTO_THRESHOLD,
    auto_activate_margin=TOOL_ROUTING_AUTO_MARGIN,
    min_candidate_score=TOOL_ROUTING_MIN_SCORE,
)
if RECIPES_ENABLED:
    init_recipe_store()
    if RECIPE_CFG.get("seed_builtin_compatibility", True):
        seed_builtin_recipes()
WORKING_STATE = WorkingStateStore(limits=WORKING_STATE_CFG)

LOG_THINKING_TRACE = False
