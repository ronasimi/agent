from tools.self_optimization import run_self_optimization_job
from ..config import CONFIG
from ..handlers import JobHandler
from ..resources import InferenceDeferred, _ensure_interactive_idle, _memory_available_mb

def _run(job_id: str, worker_id: str):
    required_mb = int(CONFIG.get("self_optimization", {}).get("min_available_memory_mb", 2200))
    available_mb = _memory_available_mb()
    if available_mb is not None and available_mb < required_mb:
        raise InferenceDeferred(f"Self-optimization needs {required_mb} MiB available RAM; detected {available_mb} MiB.")
    return run_self_optimization_job(job_id, worker_id, before_inference=_ensure_interactive_idle)

JOB_HANDLER = JobHandler("self_optimization", _run)
