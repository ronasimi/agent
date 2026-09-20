"""Localhost-only Laya sidecar used by all harness processes."""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from tools.config import load_config
from .decision_engine import LocalLayaEngine

CFG = dict((load_config().get("agent") or {}).get("decision_engine") or {})
ENGINE = LocalLayaEngine(CFG)


def _background_load() -> None:
    ENGINE.load()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if bool(CFG.get("enabled", False)) and bool(CFG.get("preload_async", True)):
        threading.Thread(target=_background_load, daemon=True, name="laya-sidecar-preload").start()
    yield


app = FastAPI(title="Al Agent Decision Engine", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": ENGINE.loaded,
        "loaded": ENGINE.loaded,
        "loading": ENGINE.loading,
        "model": ENGINE.model_id,
        "device": ENGINE.device,
        "error": ENGINE.last_error,
    }


@app.post("/preload")
def preload() -> dict[str, Any]:
    # Do not hold the request open for a potentially multi-second cold load.
    # If a cancelled cold load is still winding down, calling load() here is
    # intentionally cheap: it only marks the new desired generation so that
    # the in-flight loader retries after discarding its stale weights.
    if not ENGINE.loaded:
        if ENGINE.loading:
            ENGINE.load()
        else:
            threading.Thread(target=_background_load, daemon=True, name="laya-sidecar-preload").start()
    return {"ok": True, "loaded": ENGINE.loaded, "loading": ENGINE.loading}


@app.post("/unload")
def unload() -> dict[str, Any]:
    return {"ok": True, "unloaded": ENGINE.unload()}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    if not ENGINE.loaded:
        # Fail fast so the harness can use deterministic/2B fallback rather than
        # waiting through a cold model load on an interactive request.
        raise HTTPException(status_code=503, detail="decision engine not ready")
    state = payload.get("state") or {}
    questions = payload.get("questions") or {}
    if not isinstance(state, dict) or not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=400, detail="state and non-empty questions objects are required")
    started = time.monotonic()
    try:
        result = ENGINE.predict(state, questions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ok": True,
        "trace_id": uuid.uuid4().hex,
        "purpose": str(payload.get("purpose") or ""),
        "latency_ms": (time.monotonic() - started) * 1000.0,
        "answers": result.get("answers", {}) if isinstance(result, dict) else {},
    }
