"""Background-worker configuration shared by job handlers."""
from __future__ import annotations
import os
from tools.config import load_config

CONFIG = load_config()
AGENT_CFG = CONFIG.get("agent", {})
RESEARCH_CFG = CONFIG.get("research", {})
WORKER_CFG = CONFIG.get("worker", {})
MONITOR_CFG = CONFIG.get("host_monitor", {})
MODEL = AGENT_CFG.get("model", "agent-main:4b")
FAST_MODEL = AGENT_CFG.get("fast_model", MODEL)
VISION_MODEL = str(AGENT_CFG.get("vision_model") or MODEL)
FAST_MODEL_KEEP_ALIVE = AGENT_CFG.get("fast_model_keep_alive", 0)
REPORT_MODEL = AGENT_CFG.get("report_model", MODEL)
REPORT_MODEL_KEEP_ALIVE = AGENT_CFG.get("report_model_keep_alive", "10m")
MAIN_OPTIONS = AGENT_CFG.get("main_options", {"num_ctx": 16384, "temperature": 0.4})
FAST_OPTIONS = AGENT_CFG.get("fast_options", {"num_ctx": 16384, "temperature": 0.1, "top_p": 0.95, "top_k": 20})
REPORT_OPTIONS = AGENT_CFG.get("report_options") or {"num_ctx": 8192, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
REPORT_RESTORE_MODELS = bool(AGENT_CFG.get("report_restore_models_after_stage", True))
COMPACTION_MODEL = str(AGENT_CFG.get("compaction_model") or MODEL)
_COMPACTION_OVERRIDES = AGENT_CFG.get("compaction_options") or dict(MAIN_OPTIONS)
# Ollama keys a loaded runner by its context size, so requesting the *same*
# model with a smaller num_ctx unloads and reloads it. When compaction reuses
# the interactive model, keep the interactive context size so a background
# compaction never evicts the warm foreground model and forces the next user
# turn to pay a full load plus a full prefill.
COMPACTION_OPTIONS = dict(_COMPACTION_OVERRIDES)
if COMPACTION_MODEL == MODEL and MAIN_OPTIONS.get("num_ctx"):
    COMPACTION_OPTIONS["num_ctx"] = MAIN_OPTIONS["num_ctx"]
COMPACTION_KEEP_ALIVE = FAST_MODEL_KEEP_ALIVE if COMPACTION_MODEL == FAST_MODEL else -1
OLLAMA_HOST = AGENT_CFG.get("host", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
POLL_SECONDS = float(WORKER_CFG.get("poll_interval_seconds", 3))
HEARTBEAT_SECONDS = float(WORKER_CFG.get("heartbeat_seconds", 15))
COMPACTION_TIMEOUT_SECONDS = max(1.0, float(WORKER_CFG.get(
    "compaction_timeout_seconds",
    AGENT_CFG.get("model_transport", {}).get("timeout_seconds", 120),
)))
STALE_SECONDS = int(WORKER_CFG.get("stale_job_seconds", 180))
MAX_JOB_RUNTIME_SECONDS = max(60.0, float(WORKER_CFG.get("max_job_runtime_seconds", 7200)))
INTERACTIVE_COOLDOWN = float(WORKER_CFG.get("interactive_cooldown_seconds", 10))
MIN_AVAILABLE_RAM_MB = int(WORKER_CFG.get("min_available_memory_mb", 900))
MAX_AGENT_VRAM_MB = int(WORKER_CFG.get("max_agent_vram_mb", 7200))
MONITOR_INTERVAL = float(MONITOR_CFG.get("interval_seconds", 60))
MAINTENANCE_INTERVAL = float(WORKER_CFG.get("maintenance_interval_seconds", 21600))
