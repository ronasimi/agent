"""Research orchestration helpers using the durable research queue."""
from __future__ import annotations

import json
import os
import re

from ollama import Client

from .config import load_config
from .job_tools import enqueue_research
from .reminders import schedule_reminder

CONFIG = load_config()
FAST_MODEL = CONFIG.get("agent", {}).get("fast_model", "agent-main:2b")
FAST_OPTIONS = CONFIG.get("agent", {}).get("fast_options", {"num_ctx": 16384, "temperature": 0.1, "top_p": 0.95, "top_k": 20})
FAST_KEEP_ALIVE = CONFIG.get("agent", {}).get("fast_model_keep_alive", CONFIG.get("worker", {}).get("fast_model_keep_alive", 0))


def decompose_research_goal(research_goal: str) -> list[str]:
    """Generate focused search queries with a structured Ollama response."""
    schema = {
        "type": "object",
        "properties": {"queries": {"type": "array", "items": {"type": "string"}, "maxItems": 5}},
        "required": ["queries"],
    }
    prompt = f"Break this research goal into 3-5 complementary web searches. Goal: {research_goal}"
    try:
        client = Client(host=os.environ.get("OLLAMA_HOST", CONFIG.get("agent", {}).get("host", "http://localhost:11434")))
        response = client.generate(model=FAST_MODEL, prompt=prompt, format=schema, options=FAST_OPTIONS, keep_alive=FAST_KEEP_ALIVE, think=False)
        
        raw = response.get("response", "{}").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        payload = json.loads(raw)
        
        queries = [str(q).strip() for q in payload.get("queries", []) if str(q).strip()]
        return queries or [research_goal]
    except Exception:
        return [research_goal]


def orchestrate_research(research_goal: str) -> str:
    """Queue durable research; use get_research_status() later for progress and result path."""
    result = json.loads(enqueue_research(research_goal))
    return json.dumps({
        "job_id": result.get("job_id"),
        "status": result.get("status"),
        "message": "Research has been queued in the durable worker. Query get_research_status(job_id) for progress.",
    }, indent=2)


def compare_research_findings(goal1: str, goal2: str) -> str:
    """Queue two independent research jobs and return both durable job IDs."""
    one = json.loads(enqueue_research(goal1))
    two = json.loads(enqueue_research(goal2))
    return json.dumps({"goal1": one, "goal2": two}, indent=2)


def schedule_research_reminder(research_topic: str, days_ahead: int = 7) -> str:
    """Schedule a desktop reminder to revisit a research topic."""
    from datetime import datetime, timedelta, timezone
    when = datetime.now(timezone.utc) + timedelta(days=max(1, int(days_ahead)))
    return schedule_reminder(
        title=f"Research reminder: {research_topic}",
        message=f"Revisit the research topic: {research_topic}",
        when=when.isoformat(),
        repeat="once",
    )


def batch_research(topics: list) -> str:
    """Queue multiple durable research jobs and return their IDs."""
    jobs = [json.loads(enqueue_research(str(topic))) for topic in topics[:20]]
    return json.dumps(jobs, indent=2)


def trending_analysis(base_topic: str) -> str:
    """Queue research for a topic's current developments rather than running unbounded synchronous work."""
    return orchestrate_research(f"Current trends and recent developments in {base_topic}")


def research_with_constraints(goal: str, constraints: dict | None = None) -> str:
    """Queue research with textual constraints appended to the research goal."""
    constraints = constraints or {}
    suffix = ", ".join(f"{key}={value}" for key, value in constraints.items())
    enhanced = f"{goal} ({suffix})" if suffix else goal
    return orchestrate_research(enhanced)
