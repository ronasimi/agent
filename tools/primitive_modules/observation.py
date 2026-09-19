from __future__ import annotations

from .common import *  # noqa: F403
from .common import _json, _bounded_int, _safe_workspace, _source_text, _load_json, _get_path

def observation_get(observation_id: str) -> str:
    """Read one durable harness observation by identifier."""
    try:
        from .memory import read_observation
        return read_observation(observation_id)
    except Exception as exc:return f"Error: observation_get failed: {exc}"

def observation_compare(old_id: str, new_id: str) -> str:
    """Compare two durable observations using the harness observation diff primitive."""
    try:
        from .observation_tools import diff_observations
        return diff_observations(old_id,new_id)
    except Exception as exc:return f"Error: observation_compare failed: {exc}"
