"""Tiny Ollama-backed completion gate for the tool loop.

The 0.8B model answers one bounded question: is the user's task already fully
satisfied by the current tool-loop state?  ``true`` may bypass the 2B validator;
``false``, malformed output, or any model failure falls through to the existing
2B validator.  The micro model never selects tools, authorizes mutations,
classifies grounding, or declares a task blocked.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any


def _completion_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"complete": {"type": "boolean"}},
        "required": ["complete"],
        "additionalProperties": False,
    }


def _parse_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    raise ValueError("micro validator did not return a JSON object")


def validate_completion_with_model(
    client: Any,
    model: str,
    request: str,
    transcript: str,
    *,
    signal: dict[str, Any] | None = None,
    stage: str = "stall",
    options: dict[str, Any] | None = None,
    keep_alive: int | str = 0,
) -> dict[str, Any] | None:
    """Return a ``finish`` report only when the micro model says complete.

    The caller must already have checked deterministic requirements/grounding.
    Every non-``true`` outcome intentionally returns ``None`` so the normal 2B
    validator remains the recovery/blocking authority.
    """
    started = time.monotonic()
    try:
        response = client.generate(
            model=model,
            system=(
                "You are a completion classifier, not an assistant. Return only JSON. "
                "Set complete=true only when the user's request is already fully satisfied "
                "by the observed tool-loop state and no further action is useful. Otherwise false."
            ),
            prompt=(
                f"REQUEST:\n{str(request)[:900]}\n\n"
                f"STAGE: {stage}\n"
                f"SIGNAL: {json.dumps(signal or {}, ensure_ascii=False, default=str)[:400]}\n\n"
                f"RECENT TOOL LOOP:\n{str(transcript or '')[-1400:]}"
            ),
            format=_completion_schema(),
            options=dict(options or {}),
            keep_alive=keep_alive,
            think=False,
        )
        raw = response.get("response", "{}") if isinstance(response, dict) else getattr(response, "response", "{}")
        payload = _parse_json_object(raw)
        if payload.get("complete") is not True:
            return None
        return {
            "decision": "finish",
            "suggested_tool": "",
            "diagnosis": "task_complete",
            "micro_model_latency_ms": (time.monotonic() - started) * 1000.0,
            "validator": "micro_model",
        }
    except Exception:
        return None


class MicroValidator:
    """Process-local 0.8B completion classifier using the existing Ollama server."""

    def __init__(self, client: Any, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.client = client
        self.enabled = bool(cfg.get("enabled", True))
        self.model = str(cfg.get("model", "agent-micro:0.8b") or "agent-micro:0.8b")
        self.keep_alive = cfg.get("keep_alive", "2m")
        self.options = dict(
            cfg.get("options")
            or {"num_ctx": 2048, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "num_predict": 32}
        )

    def validate_loop(
        self,
        request: str,
        transcript: str,
        *,
        signal: dict[str, Any] | None = None,
        candidate_tools: list[str] | None = None,
        stage: str = "stall",
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        # candidate_tools is accepted for call-site compatibility but omitted
        # from the prompt: exact recovery/tool selection belongs to the 2B path.
        _ = candidate_tools
        return validate_completion_with_model(
            self.client,
            self.model,
            request,
            transcript,
            signal=signal,
            stage=stage,
            options=self.options,
            keep_alive=self.keep_alive,
        )

    def health(self) -> dict[str, Any]:
        return {
            "ok": self.enabled,
            "enabled": self.enabled,
            "provider": "ollama",
            "model": self.model,
        }
