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
FAST_MODEL = AGENT_CFG.get("fast_model", MODEL)
FAST_MODEL_KEEP_ALIVE = AGENT_CFG.get("fast_model_keep_alive", 0)
VISION_MODEL = str(AGENT_CFG.get("vision_model") or MODEL)
VISION_MODEL_KEEP_ALIVE = AGENT_CFG.get("vision_model_keep_alive", -1 if VISION_MODEL == MODEL else "2m")
MAIN_OPTIONS = dict(AGENT_CFG.get("main_options") or {"num_ctx": 16384, "temperature": 0.6, "top_p": 0.95, "top_k": 20})
FAST_OPTIONS = dict(AGENT_CFG.get("fast_options") or {"num_ctx": 16384, "temperature": 0.1, "top_p": 0.95, "top_k": 20})
# Ollama keys resident runners by model *and* context size.  If the fast role
# reuses the interactive model, align its context before building validator
# options so validator calls cannot evict/reload the warm main runner.
if MODEL == FAST_MODEL and MAIN_OPTIONS.get("num_ctx"):
    FAST_OPTIONS["num_ctx"] = MAIN_OPTIONS["num_ctx"]
VISION_OPTIONS = dict(AGENT_CFG.get("vision_options") or MAIN_OPTIONS)
# Reusing the main model for vision must reuse the exact resident runner/context
# rather than forcing Ollama to create a second runner for the same model.
if VISION_MODEL == MODEL:
    VISION_OPTIONS = {**MAIN_OPTIONS, **VISION_OPTIONS}
    if MAIN_OPTIONS.get("num_ctx"):
        VISION_OPTIONS["num_ctx"] = MAIN_OPTIONS["num_ctx"]
MAX_TOOLS_PER_TURN = max(8, int(AGENT_CFG.get("max_tools_per_turn", 12)))
REQUIREMENT_LED_SCHEMA_ONLY = bool(AGENT_CFG.get("requirement_led_schema_only", True))
STRUCTURED_PLAN_CFG = AGENT_CFG.get("structured_plan", {})
STRUCTURED_PLAN_ENABLED = bool(STRUCTURED_PLAN_CFG.get("enabled", True))
STRUCTURED_PLAN_MIN_CHARS = max(256, int(STRUCTURED_PLAN_CFG.get("min_chars", 900)))
STRUCTURED_PLAN_MIN_COMMANDS = max(2, int(STRUCTURED_PLAN_CFG.get("min_commands", 3)))
STRUCTURED_PLAN_MAX_STEPS = max(2, min(int(STRUCTURED_PLAN_CFG.get("max_steps", 96)), 128))
STRUCTURED_PLAN_MAX_TOOLS = max(1, min(int(STRUCTURED_PLAN_CFG.get("max_tools_per_step", 3)), 6))
OLLAMA_HOST = AGENT_CFG.get("host", "http://127.0.0.1:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
MAX_CTX = int(AGENT_CFG.get("context", {}).get("num_ctx", MAIN_OPTIONS.get("num_ctx", 16384)))
RESERVE_TOKENS = int(AGENT_CFG.get("context", {}).get("reserve_tokens", 1280))
RECENT_MESSAGES = int(AGENT_CFG.get("context", {}).get("recent_messages", 12))
COMPACT_AT = int(AGENT_CFG.get("context", {}).get("compact_at_tokens", max(8000, int(MAX_CTX * 0.62))))
SUMMARY_KEEP_MESSAGES = int(AGENT_CFG.get("context", {}).get("summary_keep_messages", 8))
MAX_TOOL_OUTPUT = int(AGENT_CFG.get("context", {}).get("max_tool_output_chars", 5000))
TOOL_LOOP_RESERVE = int(AGENT_CFG.get("context", {}).get("tool_loop_reserve_tokens", 4096))
# Kept as a compatibility knob for older configs/callers.  Context assembly no
# longer allows this setting to move harness system state after user history:
# strict Qwen/Ollama templates require one leading system message.  Default to
# false so any older call path that still consults the flag chooses safety over
# the former volatile-prefix optimization.
VOLATILE_CONTEXT_LAST = bool(AGENT_CFG.get("context", {}).get("volatile_blocks_last", False))
WARMUP_CFG = AGENT_CFG.get("warmup", {})
WARMUP_ENABLED = bool(WARMUP_CFG.get("enabled", True))
WARMUP_FAST_MODEL = WARMUP_ENABLED and bool(WARMUP_CFG.get("fast_model_prewarm", True))
WARMUP_PRIME_PREFIX = WARMUP_ENABLED and bool(WARMUP_CFG.get("prime_system_prefix", True))
MODEL_CAPABILITY_CFG = dict(AGENT_CFG.get("model_capabilities") or {})
MODEL_CAPABILITY_PROBE_ENABLED = bool(MODEL_CAPABILITY_CFG.get("enabled", True))
MODEL_CAPABILITY_PROBE_MAIN = MODEL_CAPABILITY_PROBE_ENABLED and bool(MODEL_CAPABILITY_CFG.get("probe_main_on_startup", True))
MODEL_CAPABILITY_PROBE_FAST = MODEL_CAPABILITY_PROBE_ENABLED and bool(MODEL_CAPABILITY_CFG.get("probe_fast_when_warmed", True))
MODEL_CAPABILITY_CACHE_PATH = str(MODEL_CAPABILITY_CFG.get("cache_path") or "/app/memory/model_capabilities.json")
MODEL_CAPABILITY_FORCE_PROBE = bool(MODEL_CAPABILITY_CFG.get("force_probe", False))
MODEL_CAPABILITY_IDLE_DELAY_SECONDS = max(0.0, float(MODEL_CAPABILITY_CFG.get("idle_delay_seconds", 3.0)))
MAX_ITERATIONS = int(AGENT_CFG.get("max_iterations", 12))
MAX_ITERATIONS_HARD = max(MAX_ITERATIONS, int(AGENT_CFG.get("max_iterations_hard", 32)))

TURN_SOFT_TIMEOUT_SECONDS = max(1.0, float(AGENT_CFG.get("turn_soft_timeout_seconds", 120)))
TURN_HARD_TIMEOUT_SECONDS = max(TURN_SOFT_TIMEOUT_SECONDS, float(AGENT_CFG.get("turn_hard_timeout_seconds", 180)))
MAX_MODEL_CALLS_PER_TURN = max(1, int(AGENT_CFG.get("max_model_calls_per_turn", 6)))
# Structured plans intentionally isolate one requirement at a time. They need a
# separate bounded execution budget; applying the ordinary six-call interactive
# ceiling makes any plan with >5 steps impossible to finish (one final synthesis
# call is also required). These limits apply only when the harness-owned
# scheduler is active.
STRUCTURED_PLAN_MAX_MODEL_CALLS = max(
    MAX_MODEL_CALLS_PER_TURN,
    # A tool-using scheduler step normally needs one call to emit the tool call
    # and a second call to consume the result/close the step, plus synthesis.
    min(int(STRUCTURED_PLAN_CFG.get("max_model_calls", (STRUCTURED_PLAN_MAX_STEPS * 2) + 1)), 256),
)
STRUCTURED_PLAN_MAX_ITERATIONS = max(
    MAX_ITERATIONS_HARD,
    min(int(STRUCTURED_PLAN_CFG.get("max_iterations", (STRUCTURED_PLAN_MAX_STEPS * 2) + 16)), 320),
)
STRUCTURED_PLAN_SOFT_TIMEOUT_SECONDS = max(
    TURN_SOFT_TIMEOUT_SECONDS,
    float(STRUCTURED_PLAN_CFG.get("soft_timeout_seconds", 3600)),
)
STRUCTURED_PLAN_HARD_TIMEOUT_SECONDS = max(
    STRUCTURED_PLAN_SOFT_TIMEOUT_SECONDS,
    float(STRUCTURED_PLAN_CFG.get("hard_timeout_seconds", 5400)),
)
# Repeated empty/invalid/error responses are a no-progress condition, not a
# reason to spend the entire global model-call safety budget.  Bound them
# independently so the hard budget remains a last-resort circuit breaker.
MODEL_NO_PROGRESS_MAX_RETRIES = max(1, int(AGENT_CFG.get("model_no_progress_max_retries", 2)))
MAX_VALIDATOR_CALLS_PER_TURN = max(0, int(AGENT_CFG.get("max_validator_calls_per_turn", 2)))
MAX_RECOVERY_ATTEMPTS_PER_REQUIREMENT = max(1, int(AGENT_CFG.get("max_recovery_attempts_per_requirement", 2)))
TOOL_TURN_NUM_PREDICT = max(64, int(AGENT_CFG.get("tool_turn_num_predict", 384)))
FINAL_NUM_PREDICT = max(128, int(AGENT_CFG.get("final_num_predict", 1024)))
TOOL_TURN_TEMPERATURE = max(0.0, float(AGENT_CFG.get("tool_turn_temperature", 0.2)))
# Reasoning-distilled GGUFs can occasionally finish an Ollama request with
# ``message.thinking`` populated but no user-visible content or tool call. Keep
# a separate, bounded recovery budget for that condition so it is not mistaken
# for a generic empty response. The hidden reasoning text is never promoted to
# visible output or injected back into the prompt.
REASONING_RECOVERY_CFG = dict(AGENT_CFG.get("reasoning_recovery") or {})
REASONING_RECOVERY_ENABLED = bool(REASONING_RECOVERY_CFG.get("enabled", True))
REASONING_RECOVERY_MAX_ATTEMPTS = max(0, int(REASONING_RECOVERY_CFG.get("max_attempts", 1)))
REASONING_RECOVERY_TOOL_NUM_PREDICT = max(
    TOOL_TURN_NUM_PREDICT,
    int(REASONING_RECOVERY_CFG.get("tool_num_predict", 1024)),
)
REASONING_RECOVERY_FINAL_NUM_PREDICT = max(
    FINAL_NUM_PREDICT,
    int(REASONING_RECOVERY_CFG.get("final_num_predict", 2048)),
)
SEMANTIC_MEMORY = bool(AGENT_CFG.get("semantic_memory_enabled", False))
THINKING_DEFAULT = bool(AGENT_CFG.get("thinking_default", False))
# Per-token reasoning traces are useful for terminal debugging but expensive in
# containerized/WebUI deployments because every fragment otherwise performs a
# synchronous stdout flush. Keep disabled unless explicitly requested.
LOG_THINKING_TRACE = bool(AGENT_CFG.get("log_thinking_trace", False))
SHOW_PERF_STATS = bool(AGENT_CFG.get("show_perf_stats", True))
MODEL_TRACE_CFG = AGENT_CFG.get("model_traces", {})
MODEL_TRACE_ENABLED = bool(MODEL_TRACE_CFG.get("enabled", True))
MODEL_TRACE_PATH = str(MODEL_TRACE_CFG.get("path") or "/app/memory/model_calls.jsonl")
MODEL_TRACE_MAX_BYTES = max(1024 * 1024, int(MODEL_TRACE_CFG.get("max_bytes", 268435456)))
RETHINK_CFG = AGENT_CFG.get("background_rethink", {})
RETHINK_ENABLED = bool(RETHINK_CFG.get("enabled", True))
RETHINK_MIN_TOOL_ITERATIONS = max(1, int(RETHINK_CFG.get("min_tool_iterations", 4)))
MODEL_TRANSPORT_CFG = AGENT_CFG.get("model_transport", {})
MODEL_PREFLIGHT_RETRIES = max(0, min(int(MODEL_TRANSPORT_CFG.get("preflight_retries", 1)), 4))
MODEL_RETRY_BASE_DELAY = max(0.0, float(MODEL_TRANSPORT_CFG.get("base_delay_seconds", 0.15)))
MODEL_RETRY_MAX_DELAY = max(MODEL_RETRY_BASE_DELAY, float(MODEL_TRANSPORT_CFG.get("max_delay_seconds", 0.75)))
MODEL_TRANSPORT_TIMEOUT = max(1.0, float(MODEL_TRANSPORT_CFG.get("timeout_seconds", 120)))
LOOP_VALIDATOR_CFG = AGENT_CFG.get("tool_loop_validator", {})
LOOP_VALIDATOR_ENABLED = bool(LOOP_VALIDATOR_CFG.get("enabled", True))
LOOP_VALIDATOR_OPTIONS = {**FAST_OPTIONS, **(LOOP_VALIDATOR_CFG.get("options") or {})}
if MODEL == FAST_MODEL and MAIN_OPTIONS.get("num_ctx"):
    LOOP_VALIDATOR_OPTIONS["num_ctx"] = MAIN_OPTIONS["num_ctx"]
LOOP_VALIDATOR_MAX_CHARS = int(LOOP_VALIDATOR_CFG.get("max_transcript_chars", 12000))
LOOP_VALIDATOR_KEEP_ALIVE = LOOP_VALIDATOR_CFG.get("keep_alive", FAST_MODEL_KEEP_ALIVE)
STALL_VALIDATOR_AFTER = max(2, int(LOOP_VALIDATOR_CFG.get("failed_step_attempts", 3)))
STALL_VALIDATOR_MAX_INTERVENTIONS = max(1, int(LOOP_VALIDATOR_CFG.get("max_interventions_per_turn", 3)))
LOOP_VALIDATOR_MAX_TOOLS = max(MAX_TOOLS_PER_TURN, int(LOOP_VALIDATOR_CFG.get("max_candidate_tools", 24)))
MAX_TOOL_CALLS_PER_ITERATION = max(1, int(AGENT_CFG.get("max_tool_calls_per_iteration", 3)))
MAX_MUTATING_CALLS_PER_ITERATION = max(1, int(AGENT_CFG.get("max_mutating_calls_per_iteration", 1)))
MAX_PARALLEL_READONLY_TOOLS = max(1, min(int(AGENT_CFG.get("max_parallel_readonly_tools", 3)), MAX_TOOL_CALLS_PER_ITERATION))
VISION_CFG = AGENT_CFG.get("vision", {})
VISION_SUPPORTS_IMAGES = bool(VISION_CFG.get("supports_images", True))
VISION_SIDECAR_WHEN_DISTINCT = bool(VISION_CFG.get("sidecar_when_distinct", True))
VISION_MAX_OBSERVATION_CHARS = max(1000, int(VISION_CFG.get("max_observation_chars", 5000)))
AUTO_ATTACH_TOOL_MEDIA = bool(VISION_CFG.get("auto_attach_tool_media", True)) and VISION_SUPPORTS_IMAGES
MAX_TOOL_MEDIA_PER_TURN = max(1, int(VISION_CFG.get("max_images_per_turn", 4)))
MAX_MEDIA_BYTES = max(262144, int(VISION_CFG.get("max_image_bytes", 4 * 1024 * 1024)))
SHARED_CTX_CFG = AGENT_CFG.get("model_context_sharing", {})
SHARED_CTX_ENABLED = bool(SHARED_CTX_CFG.get("enabled", True))
SHARED_CTX_MAX_CHARS = max(2000, int(SHARED_CTX_CFG.get("max_chars", 6000)))
WORKING_STATE_CFG = AGENT_CFG.get("working_state", {})
WORKING_STATE_ENABLED = bool(WORKING_STATE_CFG.get("enabled", True))
WORKING_STATE_HISTORY_TURNS = max(1, int(WORKING_STATE_CFG.get("main_history_turns", 1)))
WORKING_STATE_RAW_TOOL_RESULTS = max(1, int(WORKING_STATE_CFG.get("raw_tool_results", 2)))
WORKING_STATE_ARCHIVED_TOOL_RESULTS = max(0, int(WORKING_STATE_CFG.get("archived_tool_results", 4)))
WORKING_STATE_ARCHIVED_PREVIEW_CHARS = max(40, min(int(WORKING_STATE_CFG.get("archived_tool_preview_chars", 180)), 600))
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
RECIPE_FAST_PARAMETER_INFERENCE = bool(RECIPE_CFG.get("fast_parameter_inference", True))
RECIPE_FAST_PARAMETER_MIN_STAGES = max(2, int(RECIPE_CFG.get("fast_parameter_min_stages", 2)))
RECIPE_FAST_PARAMETER_MAX_CALLS = max(0, min(int(RECIPE_CFG.get("fast_parameter_max_calls_per_turn", 1)), 2))
INFERENCE_LOCK_PATH = os.environ.get("AGENT_INFERENCE_LOCK", "/app/workspace/.agent_inference.lock")

OLLAMA = Client(host=OLLAMA_HOST, timeout=MODEL_TRANSPORT_TIMEOUT)
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
