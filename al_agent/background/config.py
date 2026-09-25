"""Background-worker configuration shared by job handlers."""

from __future__ import annotations
import os
from tools.config import load_config

CONFIG = load_config()
AGENT_CFG = CONFIG.get("agent", {})
RESEARCH_CFG = CONFIG.get("research", {})
WORKER_CFG = CONFIG.get("worker", {})
MONITOR_CFG = CONFIG.get("host_monitor", {})
# Legacy exported names remain for job-provider compatibility only.
MODEL = AGENT_CFG["model"]
EXECUTOR_MODEL = DECISION_MODEL = REASONING_MODEL = FAST_MODEL = VISION_MODEL = (
    REPORT_MODEL
) = COMPACTION_MODEL = MODEL
MAIN_OPTIONS = dict(AGENT_CFG["main_options"])
FAST_OPTIONS = dict(MAIN_OPTIONS)
DECISION_OPTIONS = dict(MAIN_OPTIONS)
REASONING_OPTIONS = dict(MAIN_OPTIONS)
VISION_OPTIONS = dict(MAIN_OPTIONS)
REPORT_OPTIONS = dict(MAIN_OPTIONS)
COMPACTION_OPTIONS = dict(MAIN_OPTIONS)
FAST_MODEL_KEEP_ALIVE = AGENT_CFG.get("keep_alive", -1)
VISION_MODEL_KEEP_ALIVE = DECISION_MODEL_KEEP_ALIVE = REASONING_MODEL_KEEP_ALIVE = (
    REPORT_MODEL_KEEP_ALIVE
) = COMPACTION_KEEP_ALIVE = FAST_MODEL_KEEP_ALIVE
REPORT_RESTORE_MODELS = False
MODEL_TRANSPORT_TIMEOUT = float(
    AGENT_CFG.get("model_transport", {}).get("timeout_seconds", 60)
)
OLLAMA_HOST = AGENT_CFG.get(
    "host", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
)
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
POLL_SECONDS = float(WORKER_CFG.get("poll_interval_seconds", 3))
HEARTBEAT_SECONDS = float(WORKER_CFG.get("heartbeat_seconds", 15))
COMPACTION_TIMEOUT_SECONDS = max(
    1.0,
    float(
        WORKER_CFG.get(
            "compaction_timeout_seconds",
            AGENT_CFG.get("model_transport", {}).get("timeout_seconds", 120),
        )
    ),
)
STALE_SECONDS = int(WORKER_CFG.get("stale_job_seconds", 180))
MAX_JOB_RUNTIME_SECONDS = max(
    60.0, float(WORKER_CFG.get("max_job_runtime_seconds", 7200))
)
DURABLE_COMPUTE_QUANTUM = max(
    1, min(int(WORKER_CFG.get("durable_compute_quantum", 10000)), 100000)
)
DURABLE_COMPUTE_YIELD_DELAY_SECONDS = max(
    0.0, float(WORKER_CFG.get("durable_compute_yield_delay_seconds", 1.0))
)
INTERACTIVE_COOLDOWN = float(WORKER_CFG.get("interactive_cooldown_seconds", 10))
MIN_AVAILABLE_RAM_MB = int(WORKER_CFG.get("min_available_memory_mb", 900))
MAX_AGENT_VRAM_MB = int(WORKER_CFG.get("max_agent_vram_mb", 7200))
MONITOR_INTERVAL = float(MONITOR_CFG.get("interval_seconds", 60))
MAINTENANCE_INTERVAL = float(WORKER_CFG.get("maintenance_interval_seconds", 21600))
