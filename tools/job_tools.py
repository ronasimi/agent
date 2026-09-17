"""Agent-visible durable job management tools."""
from __future__ import annotations

import json

from .runtime import cancel_job, create_job, get_job, list_jobs


def enqueue_research(topic: str = "", priority: int = 0) -> str:
    """Queue a durable deep-research job that continues after the CLI exits or restarts."""
    topic = str(topic).strip()
    if not topic:
        return "Error: Missing required 'topic' parameter."
    job_id = create_job(
        "research",
        f"Research: {topic}",
        payload={"topic": topic},
        priority=max(-10, min(int(priority), 10)),
        max_attempts=3,
    )
    return json.dumps({"job_id": job_id, "status": "pending", "topic": topic}, indent=2)


def get_research_status(job_id: str = "") -> str:
    """Get detailed status, checkpoint state, attempts, errors, and result path for a research job."""
    if not str(job_id).strip():
        return "Error: Missing required 'job_id' parameter."
    job = get_job(job_id)
    if not job:
        return f"Job '{job_id}' not found."
    return json.dumps(job, ensure_ascii=False, indent=2)


def list_background_jobs(status: str = "", limit: int = 25) -> str:
    """List durable background jobs; optionally filter by pending, running, completed, failed, or cancelled."""
    allowed = {"", "pending", "running", "completed", "failed", "cancelled"}
    if status not in allowed:
        return f"Error: status must be one of {', '.join(sorted(allowed - {''}))}."
    return json.dumps(list_jobs(status=status, limit=limit), ensure_ascii=False, indent=2)


def cancel_background_job(job_id: str = "") -> str:
    """Cancel a pending or running durable background job."""
    if not str(job_id).strip():
        return "Error: Missing required 'job_id' parameter."
    return "Job cancelled." if cancel_job(job_id) else "Job could not be cancelled; it may not exist or may already be finished."
