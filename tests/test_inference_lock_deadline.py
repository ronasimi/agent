from __future__ import annotations

import pytest


def test_inference_lock_wait_has_a_deadline(monkeypatch, tmp_path):
    import al_agent.events as events

    monkeypatch.setattr(events, "INFERENCE_LOCK_PATH", str(tmp_path / "inference.lock"))
    monkeypatch.setattr(events, "INFERENCE_LOCK_TIMEOUT_SECONDS", 0.0)

    def always_busy(*args, **kwargs):
        raise BlockingIOError()

    monkeypatch.setattr(events.fcntl, "flock", always_busy)

    with pytest.raises(TimeoutError, match="shared model inference slot"):
        events.acquire_inference_lock()
