"""Foreground-priority and resource arbitration for background jobs."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
from tools.host_tools import gpu_snapshot_dict, ollama_runtime_snapshot
from tools.notify import notify_desktop
from tools.runtime import get_monitor_state, record_monitor_state
from .config import INTERACTIVE_COOLDOWN, MAX_AGENT_VRAM_MB, MIN_AVAILABLE_RAM_MB

class InferenceDeferred(RuntimeError):
    """Signal that a background job must yield to interactive inference."""
    defer_worker = True

def _memory_available_mb() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available / 1024**2)
    except Exception:
        return None

def _gpu_vram_in_use_mb() -> float | None:
    data = gpu_snapshot_dict()
    gpus = data.get("gpus", []) if isinstance(data, dict) else []
    values = [float(g.get("vram_used_mb", 0)) for g in gpus if isinstance(g, dict)]
    return sum(values) if values else None

def _ollama_vram_in_use_mb() -> float | None:
    try:
        raw = json.loads(ollama_runtime_snapshot())
        values = []
        for model in raw.get("models", []) if isinstance(raw, dict) else []:
            value = model.get("size_vram")
            if isinstance(value, (int, float)):
                values.append(float(value) / 1024**2)
        return sum(values) if values else None
    except Exception:
        return None

def resources_available() -> tuple[bool, str]:
    available = _memory_available_mb()
    if available is not None and available < MIN_AVAILABLE_RAM_MB:
        return False, f"Host available RAM is only {available} MiB (minimum {MIN_AVAILABLE_RAM_MB} MiB)."

    gpu_used = _gpu_vram_in_use_mb()
    if gpu_used is not None and gpu_used > MAX_AGENT_VRAM_MB:
        return False, f"Detected GPU VRAM usage of {gpu_used:.0f} MiB, over configured limit {MAX_AGENT_VRAM_MB} MiB."

    ollama_used = _ollama_vram_in_use_mb()
    if ollama_used is not None and ollama_used > MAX_AGENT_VRAM_MB:
        return False, f"Ollama reports {ollama_used:.0f} MiB VRAM currently resident, over configured limit {MAX_AGENT_VRAM_MB} MiB."
    return True, "resources available"

def _interactive_recent() -> bool:
    stamp = get_monitor_state("agent.last_interaction")
    if not stamp:
        return False
    try:
        from datetime import datetime, timezone
        last = datetime.fromisoformat(stamp)
        return (datetime.now(timezone.utc) - last).total_seconds() < INTERACTIVE_COOLDOWN
    except Exception:
        return False

def _interactive_busy() -> bool:
    """Return True while a foreground turn is active *or waiting* for inference."""
    active_turns = get_monitor_state("agent.foreground_turns", {})
    for token, turn in list(active_turns.items()):
        try:
            os.kill(int(turn["pid"]), 0)
            return True
        except (OSError, ValueError, TypeError, KeyError):
            from tools.runtime import set_foreground_turn
            set_foreground_turn(token, None)
    for key in ("agent.interaction_active", "agent.interaction_waiting"):
        active = get_monitor_state(key, False)
        if isinstance(active, dict) and active.get("pid"):
            try:
                os.kill(int(active["pid"]), 0)
                return True
            except (OSError, ValueError, TypeError):
                record_monitor_state(key, False)
    return _interactive_recent()

def _ensure_interactive_idle() -> None:
    """Guard every background Ollama request, not just each research phase."""
    if _interactive_busy():
        raise InferenceDeferred("Interactive inference is active; background model work deferred.")

def _notify(title: str, message: str) -> None:
    try:
        notify_desktop(title, message)
    except Exception:
        pass
