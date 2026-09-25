"""Single-model configuration and long-lived services."""

from __future__ import annotations

import os

from ollama import Client

from tools.config import config_path, load_config
from tools.memory import init_db
from tools.recipe_compat import seed_builtin_recipes
from tools.recipe_store import init_recipe_store
from tools.working_state import WorkingStateStore

CONFIG_PATH = str(config_path())
CONFIG = load_config()
AGENT_CFG = CONFIG["agent"]
MODEL = AGENT_CFG["model"]
MAIN_OPTIONS = dict(AGENT_CFG["main_options"])
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
MAX_TOOL_OUTPUT = int(CONTEXT_CFG.get("max_tool_output_chars", 6000))
MAX_MODEL_CALLS_PER_TURN = int(AGENT_CFG.get("max_model_calls_per_turn", 24))
MAX_TOOL_CALLS_PER_ITERATION = int(AGENT_CFG.get("max_tool_calls_per_iteration", 6))
MODEL_NO_PROGRESS_MAX_RETRIES = int(AGENT_CFG.get("model_no_progress_max_retries", 3))
TURN_HARD_TIMEOUT_SECONDS = float(AGENT_CFG.get("turn_hard_timeout_seconds", 600))
MODEL_TRANSPORT_CFG = AGENT_CFG.get("model_transport", {})
MODEL_TRANSPORT_TIMEOUT = float(MODEL_TRANSPORT_CFG.get("timeout_seconds", 60))
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
if RECIPES_ENABLED:
    init_recipe_store()
    if RECIPE_CFG.get("seed_builtin_compatibility", True):
        seed_builtin_recipes()
WORKING_STATE = WorkingStateStore(limits=WORKING_STATE_CFG)

LOG_THINKING_TRACE = False
