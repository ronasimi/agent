"""Background job handler discovery and dispatch."""
from __future__ import annotations

import importlib
import pkgutil

from .types import JobHandler


def discover_job_handlers() -> dict[str, JobHandler]:
    """Discover ``JOB_HANDLER`` exports from ``job_providers`` modules."""
    from . import job_providers
    prefix = job_providers.__name__ + "."
    result: dict[str, JobHandler] = {}
    for info in sorted(pkgutil.iter_modules(job_providers.__path__), key=lambda item: item.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(prefix + info.name)
        handler = getattr(module, "JOB_HANDLER", None)
        if not isinstance(handler, JobHandler):
            raise RuntimeError(f"{module.__name__} does not export a valid JOB_HANDLER")
        if handler.name in result:
            raise RuntimeError(f"Duplicate background job handler '{handler.name}'")
        result[handler.name] = handler
    return result


JOB_HANDLERS = discover_job_handlers()


def allowed_job_types() -> list[str]:
    return list(JOB_HANDLERS)


def run_job(job_type: str, job_id: str, worker_id: str):
    handler = JOB_HANDLERS.get(str(job_type))
    if not handler:
        raise RuntimeError(f"No background handler registered for job type '{job_type}'.")
    return handler.run(job_id, worker_id)
