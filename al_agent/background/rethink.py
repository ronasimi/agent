"""Low-priority post-turn reflection using the already-resident fast model."""
from __future__ import annotations

import json
import os
import re
from ollama import Client

from tools.reflection import store_reflection_notes
from tools.runtime import complete_job, get_job
from .config import FAST_MODEL, FAST_MODEL_KEEP_ALIVE, FAST_OPTIONS, OLLAMA_HOST
from .resources import _ensure_interactive_idle


def _parse_json_object(text: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.S | re.I)
    if fence:
        raw = fence.group(1)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def run_rethink_job(job_id: str) -> str:
    job = get_job(job_id)
    if not job:
        raise RuntimeError(f"Job {job_id} was not found.")
    payload = job.get("payload") or {}
    conversation_id = str(payload.get("conversation_id") or "default")
    transcript = payload.get("transcript") or []
    prompt = (
        "Review this completed assistant turn only for reusable operational lessons. "
        "Do not add world facts, user identity claims, or speculative preferences. "
        "Return strict JSON: {\"notes\":[{\"trigger_terms\":[\"term\"],\"note\":\"procedural lesson\"}]}. "
        "Use 0-3 notes. A note must be grounded in an observed tool failure/recovery, user correction, or repeated workflow issue. "
        "If nothing reusable happened, return {\"notes\":[]}.\n\n"
        + json.dumps(transcript, ensure_ascii=False, default=str)[:18000]
    )
    _ensure_interactive_idle()
    options = dict(FAST_OPTIONS)
    options["num_predict"] = min(int(options.get("num_predict") or 192), 192)
    options["temperature"] = min(float(options.get("temperature") or 0.1), 0.15)
    response = Client(host=os.environ.get("OLLAMA_HOST", OLLAMA_HOST), timeout=45).generate(
        model=FAST_MODEL,
        prompt=prompt,
        options=options,
        keep_alive=FAST_MODEL_KEEP_ALIVE,
        think=False,
    )
    parsed = _parse_json_object(response.get("response", "") if isinstance(response, dict) else getattr(response, "response", ""))
    stored = store_reflection_notes(conversation_id, parsed.get("notes") if isinstance(parsed.get("notes"), list) else [])
    result = f"Stored {stored} bounded reflection note(s)."
    complete_job(job_id, result)
    return result
