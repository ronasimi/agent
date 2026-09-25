"""Cross-process inference arbitration for the single resident model."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager

INFERENCE_LOCK_PATH = os.environ.get(
    "AGENT_INFERENCE_LOCK", "/app/workspace/.agent_inference.lock"
)


@contextmanager
def background_inference_slot():
    """Defer background inference immediately while foreground work is active."""
    from .background.resources import InferenceDeferred, _ensure_interactive_idle

    _ensure_interactive_idle()
    os.makedirs(os.path.dirname(INFERENCE_LOCK_PATH) or ".", exist_ok=True)
    with open(INFERENCE_LOCK_PATH, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InferenceDeferred(
                "Foreground inference owns the model queue"
            ) from exc
        try:
            _ensure_interactive_idle()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
