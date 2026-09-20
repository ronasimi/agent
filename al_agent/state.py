"""Process-wide immutable configuration and long-lived service objects."""
from __future__ import annotations

import os
from ollama import Client

from tools.config import load_config
from tools.memory import init_db
from tools.recipe_store import init_recipe_store
from tools.recipe_compat import seed_builtin_recipes
from tools.working_state import WorkingStateStore

CONFIG_PATH = os.environ.get("AGENT_CONFIG", "/app/config/config.yaml")
CONFIG = load_config()
AGENT_CFG = CONFIG.get("agent", {})
MODEL = AGENT_CFG.get("model", "agent-main:4b")
FAST_MODEL = AGENT_CFG.get("fast_model", "agent-fast:2b")
FAST_MODEL_KEEP_ALIVE = AGENT_CFG.get("fast_model_keep_alive", 0)
MAIN_OPTIONS = AGENT_CFG.get("main_options") or {"num_ctx": 16384, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
FAST_OPTIONS = AGENT_CFG.get("fast_options") or {"num_ctx": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
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
# Volatile harness blocks (working state, evidence digest) are placed after the
# stable history so a changed block does not invalidate the server-side prompt
# prefix cache for the system prompt and conversation history.
VOLATILE_CONTEXT_LAST = bool(AGENT_CFG.get("context", {}).get("volatile_blocks_last", True))
WARMUP_CFG = AGENT_CFG.get("warmup", {})
WARMUP_ENABLED = bool(WARMUP_CFG.get("enabled", True))
WARMUP_PRIME_PREFIX = WARMUP_ENABLED and bool(WARMUP_CFG.get("prime_system_prefix", True))
MAX_ITERATIONS = int(AGENT_CFG.get("max_iterations", 12))
MAX_ITERATIONS_HARD = max(MAX_ITERATIONS, int(AGENT_CFG.get("max_iterations_hard", 32)))
SEMANTIC_MEMORY = bool(AGENT_CFG.get("semantic_memory_enabled", False))
THINKING_DEFAULT = bool(AGENT_CFG.get("thinking_default", False))
SHOW_PERF_STATS = bool(AGENT_CFG.get("show_perf_stats", True))
MODEL_TRANSPORT_CFG = AGENT_CFG.get("model_transport", {})
MODEL_PREFLIGHT_RETRIES = max(0, min(int(MODEL_TRANSPORT_CFG.get("preflight_retries", 1)), 4))
MODEL_RETRY_BASE_DELAY = max(0.0, float(MODEL_TRANSPORT_CFG.get("base_delay_seconds", 0.15)))
MODEL_RETRY_MAX_DELAY = max(MODEL_RETRY_BASE_DELAY, float(MODEL_TRANSPORT_CFG.get("max_delay_seconds", 0.75)))
LOOP_VALIDATOR_CFG = AGENT_CFG.get("tool_loop_validator", {})
LOOP_VALIDATOR_ENABLED = bool(LOOP_VALIDATOR_CFG.get("enabled", True))
LOOP_VALIDATOR_OPTIONS = {**FAST_OPTIONS, **(LOOP_VALIDATOR_CFG.get("options") or {})}
LOOP_VALIDATOR_MAX_CHARS = int(LOOP_VALIDATOR_CFG.get("max_transcript_chars", 12000))
LOOP_VALIDATOR_KEEP_ALIVE = LOOP_VALIDATOR_CFG.get("keep_alive", FAST_MODEL_KEEP_ALIVE)
STALL_VALIDATOR_AFTER = max(2, int(LOOP_VALIDATOR_CFG.get("failed_step_attempts", 3)))
STALL_VALIDATOR_MAX_INTERVENTIONS = max(1, int(LOOP_VALIDATOR_CFG.get("max_interventions_per_turn", 3)))
LOOP_VALIDATOR_MAX_TOOLS = max(MAX_TOOLS_PER_TURN, int(LOOP_VALIDATOR_CFG.get("max_candidate_tools", 24)))
MAX_TOOL_CALLS_PER_ITERATION = max(1, int(AGENT_CFG.get("max_tool_calls_per_iteration", 3)))
MAX_MUTATING_CALLS_PER_ITERATION = max(1, int(AGENT_CFG.get("max_mutating_calls_per_iteration", 1)))
MAX_PARALLEL_READONLY_TOOLS = max(1, min(int(AGENT_CFG.get("max_parallel_readonly_tools", 3)), MAX_TOOL_CALLS_PER_ITERATION))
VISION_CFG = AGENT_CFG.get("vision", {})
AUTO_ATTACH_TOOL_MEDIA = bool(VISION_CFG.get("auto_attach_tool_media", True))
MAX_TOOL_MEDIA_PER_TURN = max(1, int(VISION_CFG.get("max_images_per_turn", 4)))
MAX_MEDIA_BYTES = max(262144, int(VISION_CFG.get("max_image_bytes", 4 * 1024 * 1024)))
SHARED_CTX_CFG = AGENT_CFG.get("model_context_sharing", {})
SHARED_CTX_ENABLED = bool(SHARED_CTX_CFG.get("enabled", True))
SHARED_CTX_MAX_CHARS = max(2000, int(SHARED_CTX_CFG.get("max_chars", 6000)))
WORKING_STATE_CFG = AGENT_CFG.get("working_state", {})
WORKING_STATE_ENABLED = bool(WORKING_STATE_CFG.get("enabled", True))
WORKING_STATE_HISTORY_TURNS = max(1, int(WORKING_STATE_CFG.get("main_history_turns", 1)))
WORKING_STATE_RAW_TOOL_RESULTS = max(1, int(WORKING_STATE_CFG.get("raw_tool_results", 2)))
WORKING_STATE_EVIDENCE_CHARS = max(600, int(WORKING_STATE_CFG.get("evidence_render_chars", 3600)))
PRUNE_SATISFIED_REQUIREMENT_TOOLS = bool(WORKING_STATE_CFG.get("prune_satisfied_tool_schemas", True))
# Tool schemas are rendered at the top of the prompt by chat templates, so any
# change to the set - or to its order - invalidates the whole server-side prefix
# cache. Prune and reorder only in an iteration that already has to expose a new
# required tool.
MINIMIZE_SCHEMA_CHURN = bool(WORKING_STATE_CFG.get("minimize_schema_churn", True))
SUPPRESS_COMPLETED_REQUIREMENT_REPEATS = bool(WORKING_STATE_CFG.get("suppress_completed_requirement_repeats", True))
REQUIREMENT_TOOL_CAP = max(MAX_TOOLS_PER_TURN, int(AGENT_CFG.get("requirement_tool_cap", 24)))
GROUNDING_CFG = AGENT_CFG.get("grounding", {})
GROUNDING_ENABLED = bool(GROUNDING_CFG.get("enabled", True))
WEATHER_GROUNDING_MAX_AGE_SECONDS = max(300, int(GROUNDING_CFG.get("weather_max_age_seconds", 10800)))
# Discarding a candidate answer produces no new evidence, so the gate needs its
# own bound or it can consume every remaining iteration of the turn.
GROUNDING_MAX_DISCARDS = max(1, int(GROUNDING_CFG.get("max_candidate_discards", 3)))
RECIPE_CFG = AGENT_CFG.get("recipes", {})
RECIPES_ENABLED = bool(RECIPE_CFG.get("enabled", True))
RECIPE_SUGGEST = bool(RECIPE_CFG.get("suggest_after_success", True))
RECIPE_MIN_STAGES = max(1, int(RECIPE_CFG.get("min_stages", 1)))
RECIPE_MATCH_THRESHOLD = float(RECIPE_CFG.get("semantic_match_threshold", 0.35))
RECIPE_PREFLIGHT_LIMIT = max(1, min(int(RECIPE_CFG.get("preflight_limit", 3)), 8))
RECIPE_VALIDATOR_FALLBACK = bool(RECIPE_CFG.get("validator_fallback_enabled", True))
RECIPE_VALIDATOR_MAX_STAGES = max(1, min(int(RECIPE_CFG.get("validator_fallback_max_stages", 4)), 8))
RECIPE_VALIDATOR_MAX_TOOLS = max(1, min(int(RECIPE_CFG.get("validator_fallback_max_tools", 12)), LOOP_VALIDATOR_MAX_TOOLS))
INFERENCE_LOCK_PATH = os.environ.get("AGENT_INFERENCE_LOCK", "/app/workspace/.agent_inference.lock")

OLLAMA = Client(host=OLLAMA_HOST)
LOOP_VALIDATOR_CLIENT = Client(host=OLLAMA_HOST, timeout=float(LOOP_VALIDATOR_CFG.get("timeout_seconds", 45)))

init_db()
if RECIPES_ENABLED:
    init_recipe_store()
    if bool(RECIPE_CFG.get("seed_builtin_compatibility", True)):
        seed_builtin_recipes()
WORKING_STATE = WorkingStateStore(limits={
    key: value for key, value in WORKING_STATE_CFG.items()
    if key not in {"enabled", "main_history_turns"}
})
