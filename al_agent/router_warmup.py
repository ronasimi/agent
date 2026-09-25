"""Idle-only router prefill maintenance; healthy models need no keepalive inference."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any


def _field(obj: Any, key: str, default=None):
    return (
        obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    )


def _canonical_model(name: str) -> str:
    name = str(name or "")
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


class RouterPrefixWarmer:
    """One Web UI process owns maintenance; foreground requests prime themselves.

    /api/ps cannot reveal every unload/reload that happens between polls. Such a
    request still sends the exact prefix and naturally repopulates Ollama's cache.
    No separate warmup request is inserted into the foreground routing path.
    """

    def __init__(
        self,
        router,
        *,
        catalog: Callable,
        resident_models: Callable,
        inference_slot: Callable = nullcontext,
        retry_seconds: float = 60,
        clock: Callable = time.monotonic,
        status_sink: Callable | None = None,
        main_model: str = "",
    ):
        self.router = router
        self.catalog = catalog
        self.resident_models = resident_models
        self.inference_slot = inference_slot
        self.retry_seconds = max(1, float(retry_seconds))
        self.clock = clock
        self.status_sink = status_sink
        self.main_model = main_model
        self._primed_key = None
        self._identity = None
        self._next_attempt = 0.0
        self._failures = 0
        self.status: dict[str, Any] = {"state": "pending", "warm_attempts": 0}

    def _status(self, **values):
        updated = {**self.status, **values}
        if updated == self.status:
            return
        self.status = updated
        if self.status_sink:
            try:
                self.status_sink(dict(updated))
            except Exception:
                pass

    def tick(self) -> bool:
        """Inspect residency/catalog and perform at most one necessary idle prefill."""
        try:
            schemas = self.catalog()
            index = self.router.build_index(schemas)
            key = hashlib.sha256(
                json.dumps(
                    [index.fingerprint, self.router.model, self.router.options],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            response = self.resident_models()
            models = _field(response, "models", []) or []
            by_name = {
                _canonical_model(_field(m, "model") or _field(m, "name", "")): m
                for m in models
            }
            resident = by_name.get(_canonical_model(self.router.model))
            identity = (
                None
                if resident is None
                else (_field(resident, "digest"), _field(resident, "context_length"))
            )
            correct_context = resident is not None and (
                _field(resident, "context_length")
                in (None, self.router.options["num_ctx"])
            )
            main_resident = (
                not self.main_model or _canonical_model(self.main_model) in by_name
            )
            self._status(
                prefix_fingerprint=index.fingerprint,
                prefix_bytes=len(index.prefix.encode()),
                catalog_tools=len(index.names),
                main_resident=main_resident,
                router_resident=resident is not None,
            )
        except Exception as exc:
            # A server restart can invalidate a prefix even if its next /ps
            # result has the same digest/context as before.
            self._primed_key = None
            self._status(state="check_failed", error=f"{type(exc).__name__}: {exc}")
            return False

        if self._primed_key == key and correct_context and self._identity == identity:
            self._status(state="resident", error="")
            return False
        if self.clock() < self._next_attempt:
            return False
        try:
            with self.inference_slot():
                self._status(
                    state="warming",
                    warm_attempts=self.status["warm_attempts"] + 1,
                    error="",
                )
                if not self.router.warm(schemas):
                    raise RuntimeError(
                        "Router prefix warmup failed; see router_prefix_warmup trace."
                    )
        except Exception as exc:
            # A busy foreground slot is not a model failure. Retry on the next
            # idle tick; genuine failures back off to avoid a load/eviction loop.
            if getattr(exc, "defer_worker", False) or isinstance(exc, BlockingIOError):
                self._status(state="deferred", error="")
            else:
                self._failures += 1
                self._next_attempt = self.clock() + min(
                    900, self.retry_seconds * 2 ** min(self._failures - 1, 4)
                )
                self._status(state="warm_failed", error=f"{type(exc).__name__}: {exc}")
            return False

        self._primed_key = key
        self._failures = 0
        # Also throttle repeated successful reloads caused by insufficient
        # memory or a host configured with MAX_LOADED_MODELS=1.
        self._next_attempt = self.clock() + self.retry_seconds
        try:
            after = _field(self.resident_models(), "models", []) or []
            current = next(
                (
                    m
                    for m in after
                    if _canonical_model(_field(m, "model") or _field(m, "name", ""))
                    == _canonical_model(self.router.model)
                ),
                None,
            )
            self._identity = (
                None
                if current is None
                else (_field(current, "digest"), _field(current, "context_length"))
            )
            self._status(
                router_resident=current is not None,
                main_resident=not self.main_model
                or any(
                    _canonical_model(_field(m, "model") or _field(m, "name", ""))
                    == _canonical_model(self.main_model)
                    for m in after
                ),
            )
        except Exception:
            self._identity = identity
        self._status(state="primed", last_warm_at=time.time(), error="")
        return True
