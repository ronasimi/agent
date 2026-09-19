"""Background-worker configuration shared by job handlers."""
from __future__ import annotations
import os
from tools.config import load_config

CONFIG = load_config()
AGENT_CFG = CONFIG.get("agent", {})
RESEARCH_CFG = CONFIG.get("research", {})
WORKER_CFG = CONFIG.get("worker", {})
MONITOR_CFG = CONFIG.get("host_monitor", {})
MODEL = AGENT_CFG.get("model", "qwen3.5:4b")
MAIN_OPTIONS = AGENT_CFG.get("main_options", {"num_ctx": 16384, "temperature": 0.4})
COMPACTION_MODEL = str(AGENT_CFG.get("compaction_model") or MODEL)
COMPACTION_OPTIONS = AGENT_CFG.get("compaction_options") or dict(MAIN_OPTIONS)
OLLAMA_HOST = AGENT_CFG.get("host", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
POLL_SECONDS = float(WORKER_CFG.get("poll_interval_seconds", 3))
HEARTBEAT_SECONDS = float(WORKER_CFG.get("heartbeat_seconds", 15))
STALE_SECONDS = int(WORKER_CFG.get("stale_job_seconds", 180))
INTERACTIVE_COOLDOWN = float(WORKER_CFG.get("interactive_cooldown_seconds", 10))
MIN_AVAILABLE_RAM_MB = int(WORKER_CFG.get("min_available_memory_mb", 900))
MAX_AGENT_VRAM_MB = int(WORKER_CFG.get("max_agent_vram_mb", 7200))
MONITOR_INTERVAL = float(MONITOR_CFG.get("interval_seconds", 60))
MAINTENANCE_INTERVAL = float(WORKER_CFG.get("maintenance_interval_seconds", 21600))
