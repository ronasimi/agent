"""Optional Laya System-1 decision layer.

The harness talks to a localhost Laya sidecar instead of loading a 421M encoder
inside every agent/web/worker process. This keeps one shared copy resident,
allows /research to explicitly unload it while the 9B writer is active, and
keeps the normal harness fail-open when the optional service is absent.

Deterministic harness rules remain authoritative. Laya may narrow read-only
schema exposure or end a validator loop at high confidence; it never grants
mutating authority and never overrides deterministic factual grounding.
"""
from __future__ import annotations

import fcntl
import gc
import inspect
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from tools.runtime import record_monitor_state, utc_now

_LAST_KEY = "agent.decision_engine_last"

# Small read-only starter sets. A classifier is never allowed to expose a
# mutating tool that deterministic/lexical policy did not already select.
_FAMILY_TOOLS: dict[str, tuple[str, ...]] = {
    "system": (
        "host_snapshot", "process_snapshot", "pressure_snapshot", "filesystem_snapshot",
        "service_health", "read_host_journal", "kernel_info", "memory_info",
    ),
    "network": (
        "network_snapshot", "dns_diagnose", "network_path", "endpoint_probe",
        "http_probe", "local_subnets", "scan_subnet", "resolve_host",
    ),
    "files": (
        "read_text", "read_lines", "list_directory", "find_paths", "text_search",
        "path_stat", "json_query", "csv_summary",
    ),
    "web": (
        "web_search", "browse_url", "news_search", "wiki_search", "page_metadata",
        "page_links", "read_feed", "extract_document",
    ),
    "memory": ("search_memory", "profile_image_info"),
    "coding": (
        "read_file", "read_text", "get_repo_map", "search_repo_symbols", "read_repo_symbol",
        "repo_status", "repo_diff", "repo_checks",
    ),
    "workflow": ("search_recipes", "list_recipes", "run_recipe", "recipe_coverage"),
    "research": ("web_search", "browse_url", "wiki_search", "news_search", "get_research_status"),
    "automation": ("list_reminders",),
    "conversation": (),
    "other": (),
}
_TOOL_TO_FAMILY = {tool: family for family, tools in _FAMILY_TOOLS.items() for tool in tools}


@dataclass
class ChoiceDecision:
    value: str = ""
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionBatch:
    ok: bool
    answers: dict[str, ChoiceDecision] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str = ""
    trace_id: str = ""
    source: str = "laya"

    def accepted(self, name: str, threshold: float) -> ChoiceDecision | None:
        answer = self.answers.get(name)
        if answer and answer.value and answer.confidence >= float(threshold):
            return answer
        return None


def route_questions() -> dict[str, Any]:
    """Batched coarse routing questions kept below Laya's option-count limits."""
    return {
        "route_family": {
            "type": "choice",
            "instructions": "Which broad capability family best handles the current user request?",
            "criteria": {
                "conversation": "ordinary conversation, explanation, or timeless general knowledge with no local/external action",
                "system": "host OS, processes, services, logs, CPU, RAM, disks, or machine diagnostics",
                "network": "DNS, routes, ports, LAN/subnets, connectivity, TLS, HTTP endpoint diagnostics",
                "files": "read, inspect, search, parse, compare, or summarize local files/directories",
                "web": "current external information, websites, news, pages, or internet retrieval",
                "memory": "recall or inspect previously stored user/conversation memory",
                "coding": "code, repositories, debugging, implementation, tests, or source inspection",
                "workflow": "recipes, pipelines, reusable workflows, or tool orchestration",
                "research": "long-running multi-source research or research-job status",
                "automation": "reminders, scheduled work, notifications, or queued jobs",
                "other": "none of the above is a clear fit",
            },
        },
        "tool_requirement": {
            "type": "choice",
            "instructions": "Does satisfying the current request require calling an external/local tool rather than answering from reasoning alone?",
            "criteria": {
                "none": "no tool is needed to answer the request correctly",
                "optional": "a tool could help but is not required",
                "required": "the request cannot be correctly completed without a tool or external/local state",
            },
        },
        "freshness": {
            "type": "choice",
            "instructions": "How time-sensitive is the factual information requested?",
            "criteria": {
                "timeless": "stable/general information; no current data requirement",
                "current": "recent or up-to-date information is requested",
                "realtime": "live/current-now state or price/status is requested",
            },
        },
        "continuation": {
            "type": "choice",
            "instructions": "Is the current request referentially continuing the previous user task?",
            "criteria": {
                "new": "independent new task",
                "continuation": "depends on the previous user's request/entity/result",
                "ambiguous": "could be either; context is insufficient",
            },
        },
        "renderer": {
            "type": "choice",
            "instructions": "If a structured tool result directly answers the request, is deterministic rendering preferable to free-form model synthesis?",
            "criteria": {
                "direct": "simple structured lookup/status/list can be rendered mechanically",
                "model": "requires explanation, synthesis, reasoning, or nuanced prose",
            },
        },
        "risk": {
            "type": "choice",
            "instructions": "What side-effect level does the user request imply?",
            "criteria": {
                "read_only": "inspection/retrieval only",
                "mutation": "explicitly asks to change local/external state",
                "mixed": "contains both inspection and changes",
                "unknown": "side-effect intent is unclear",
            },
        },
    }


def validator_questions() -> dict[str, Any]:
    return {
        "action": {
            "type": "choice",
            "instructions": "What should the harness do next based only on the request and observed tool-loop state?",
            "criteria": {
                "finish": "the user's task is already satisfied and no more tool work is needed",
                "recover": "the task remains incomplete and a corrective/different tool action is needed",
                "blocked": "the task cannot be completed safely with the available evidence/capabilities",
            },
        },
        "diagnosis": {
            "type": "choice",
            "instructions": "What best describes the current tool-loop state?",
            "criteria": {
                "task_complete": "requested work is complete",
                "bad_arguments": "tool arguments were invalid or incomplete",
                "wrong_tool": "the selected tool does not match the task",
                "transient_failure": "a temporary tool/backend failure occurred",
                "insufficient_evidence": "more evidence is needed before answering",
                "repeated_call": "the loop is repeating without useful progress",
                "tool_unavailable": "needed capability is unavailable or blocked",
                "unknown": "none of the above is clearly supported",
            },
        },
    }


def _route_json_schema() -> dict[str, Any]:
    """Constrained schema used only when Laya is low-confidence.

    The 2B fallback is deliberately categorical and tiny. It is not consulted
    when deterministic routing already resolved the turn.
    """
    return {
        "type": "object",
        "properties": {
            "route_family": {"type": "string", "enum": list(route_questions()["route_family"]["criteria"])},
            "tool_requirement": {"type": "string", "enum": list(route_questions()["tool_requirement"]["criteria"])},
            "freshness": {"type": "string", "enum": list(route_questions()["freshness"]["criteria"])},
            "continuation": {"type": "string", "enum": list(route_questions()["continuation"]["criteria"])},
            "renderer": {"type": "string", "enum": list(route_questions()["renderer"]["criteria"])},
            "risk": {"type": "string", "enum": list(route_questions()["risk"]["criteria"])},
        },
        "required": ["route_family", "tool_requirement", "freshness", "continuation", "renderer", "risk"],
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
    raise ValueError("route fallback did not return a JSON object")


def route_with_fast_model(
    client: Any,
    model: str,
    request: str,
    *,
    previous_user: str = "",
    previous_frame: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    keep_alive: int | str = 0,
) -> DecisionBatch:
    """Use the 2B model only when a Laya route is unavailable/uncertain.

    This preserves the intended hierarchy: deterministic rules -> Laya -> 2B.
    It still cannot grant mutation authority; callers apply the same read-only
    narrowing and deterministic tool-policy gates used for Laya decisions.
    """
    started = time.monotonic()
    try:
        frame = previous_frame or {}
        response = client.generate(
            model=model,
            system=(
                "You are a routing classifier, not an assistant. Classify the current user request into the supplied "
                "JSON fields. Do not solve the request and do not invent tools. The previous user request/frame are "
                "context only for deciding whether this is a continuation."
            ),
            prompt=(
                f"CURRENT REQUEST:\n{str(request)[:1800]}\n\n"
                f"PREVIOUS USER REQUEST:\n{str(previous_user)[:800]}\n\n"
                f"PREVIOUS INTENT: {str(frame.get('intent') or '')[:120]}\n"
                f"PREVIOUS ENTITY: {str(frame.get('entity') or '')[:240]}"
            ),
            format=_route_json_schema(),
            options=dict(options or {}),
            keep_alive=keep_alive,
            think=False,
        )
        raw = response.get("response", "{}") if isinstance(response, dict) else getattr(response, "response", "{}")
        payload = _parse_json_object(raw)
        criteria = route_questions()
        answers: dict[str, ChoiceDecision] = {}
        for name in _route_json_schema()["required"]:
            value = str(payload.get(name) or "")
            if value not in criteria[name]["criteria"]:
                raise ValueError(f"invalid {name}: {value}")
            # Constrained 2B output is a fallback label, not a calibrated probability.
            # Mark it accepted for routing while recording its source explicitly.
            answers[name] = ChoiceDecision(value=value, confidence=1.0)
        return DecisionBatch(
            ok=True, answers=answers, latency_ms=(time.monotonic() - started) * 1000.0,
            trace_id=uuid.uuid4().hex, source="fast_model",
        )
    except Exception as exc:
        return DecisionBatch(
            ok=False, latency_ms=(time.monotonic() - started) * 1000.0,
            error=str(exc), trace_id=uuid.uuid4().hex, source="fast_model",
        )


class LocalLayaEngine:
    """Single-process Laya runtime used only by the localhost sidecar."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.model_id = str(cfg.get("model", "convaiinnovations/laya") or "convaiinnovations/laya")
        self.subfolder = str(cfg.get("subfolder", "") or "")
        self.device = str(cfg.get("device", "cpu") or "cpu")
        self.max_state_chars = max(600, int(cfg.get("max_state_chars", 1800)))
        self._agent: Any = None
        self._lock = threading.RLock()
        self._predict_lock = threading.Lock()
        self._loading = False
        self._desired_loaded = False
        self._load_generation = 0
        self._last_error = ""

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._agent is not None

    @property
    def loading(self) -> bool:
        with self._lock:
            return self._loading

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def load(self) -> bool:
        with self._lock:
            was_desired = self._desired_loaded
            self._desired_loaded = True
            if self._agent is not None:
                return True
            if self._loading:
                # If an unload happened while a cold load was in flight, a new
                # preload request invalidates that in-flight generation. The
                # loader will discard its weights and immediately retry once.
                if not was_desired:
                    self._load_generation += 1
                return False
            self._loading = True
            generation = self._load_generation
        agent: Any = None
        ok = False
        retry = False
        try:
            import laya
            kwargs: dict[str, Any] = {}
            if self.subfolder:
                kwargs["subfolder"] = self.subfolder
            try:
                if "device" in inspect.signature(laya.load).parameters:
                    kwargs["device"] = self.device
            except (TypeError, ValueError):
                pass
            try:
                agent = laya.load(self.model_id, **kwargs)
            except TypeError:
                kwargs.pop("device", None)
                agent = laya.load(self.model_id, **kwargs)
            with self._lock:
                if self._desired_loaded and generation == self._load_generation:
                    self._agent = agent
                    self._last_error = ""
                    ok = True
                else:
                    # /research can cancel a sidecar cold load before the 9B
                    # writer is admitted. Never publish stale loaded weights.
                    agent = None
                retry = self._desired_loaded and generation != self._load_generation
            return ok
        except Exception as exc:
            with self._lock:
                self._agent = None
                self._last_error = str(exc)
            return False
        finally:
            with self._lock:
                self._loading = False
            if agent is None:
                gc.collect()
            # A preload requested after an in-flight load was cancelled should
            # eventually win without requiring a second external HTTP call.
            if retry:
                self.load()

    def unload(self) -> bool:
        # Wait for the tiny inference critical section, then prevent both an
        # already-loaded model and an in-flight cold load from becoming resident.
        with self._predict_lock:
            with self._lock:
                self._desired_loaded = False
                self._load_generation += 1
                had = self._agent is not None
                self._agent = None
        gc.collect()
        try:
            import sys
            torch = sys.modules.get("torch")
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return had

    def _clip_state(self, state: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        budget = self.max_state_chars
        for key, value in state.items():
            if budget <= 0:
                break
            text = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
                if isinstance(value, (dict, list, tuple)) else str(value or "")
            )
            take = min(len(text), budget)
            out[str(key)] = text[:take]
            budget -= take
        return out

    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        # Serialize encoder passes and make unload wait for any active pass. This
        # guarantees that report-stage memory handoff cannot overlap a hidden
        # Laya inference with the 9B writer load.
        with self._predict_lock:
            with self._lock:
                agent = self._agent
            if agent is None:
                raise RuntimeError("Laya checkpoint is not loaded")
            return agent.predict(self._clip_state(state), questions)


class DecisionEngineClient:
    """Fail-open HTTP client used by the harness and worker processes."""

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.endpoint = str(cfg.get("endpoint", "http://127.0.0.1:8091") or "http://127.0.0.1:8091").rstrip("/")
        self.model_id = str(cfg.get("model", "convaiinnovations/laya") or "convaiinnovations/laya")
        self.timeout = max(0.05, float(cfg.get("request_timeout_seconds", 1.5)))
        self.route_cfg = dict(cfg.get("routing") or {})
        self.validator_cfg = dict(cfg.get("validator") or {})
        capture = dict(cfg.get("training_capture") or {})
        self.capture_enabled = bool(capture.get("enabled", True))
        self.capture_path = str(capture.get("path", "/app/memory/laya_training.jsonl"))
        self.capture_text = bool(capture.get("include_request_text", True))

    @property
    def routing_enabled(self) -> bool:
        return self.enabled and bool(self.route_cfg.get("enabled", True))

    @property
    def validator_enabled(self) -> bool:
        return self.enabled and bool(self.validator_cfg.get("enabled", True))

    def threshold(self, name: str, default: float) -> float:
        thresholds = dict(self.route_cfg.get("thresholds") or {})
        return max(0.0, min(float(thresholds.get(name, default)), 1.0))

    @property
    def validator_threshold(self) -> float:
        return max(0.0, min(float(self.validator_cfg.get("min_confidence", 0.95)), 1.0))

    @staticmethod
    def _choice_from(raw: Any) -> ChoiceDecision:
        if not isinstance(raw, dict):
            return ChoiceDecision()
        try:
            confidence = float(raw.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        probs = raw.get("probabilities") or raw.get("distribution") or {}
        probabilities: dict[str, float] = {}
        if isinstance(probs, dict):
            for key, val in probs.items():
                try:
                    probabilities[str(key)] = float(val)
                except (TypeError, ValueError):
                    pass
        return ChoiceDecision(
            value=str(raw.get("choice") or ""),
            confidence=max(0.0, min(confidence, 1.0)),
            probabilities=probabilities,
        )

    def predict(self, state: dict[str, Any], questions: dict[str, Any], *, purpose: str) -> DecisionBatch:
        if not self.enabled:
            return DecisionBatch(ok=False, error="decision engine disabled")
        started = time.monotonic()
        trace_id = uuid.uuid4().hex
        try:
            response = requests.post(
                self.endpoint + "/predict",
                json={"state": state, "questions": questions, "purpose": purpose},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return DecisionBatch(ok=False, latency_ms=(time.monotonic() - started) * 1000, error=f"HTTP {response.status_code}", trace_id=trace_id)
            payload = response.json()
            raw_answers = payload.get("answers") or {}
            answers = {name: self._choice_from(raw_answers.get(name)) for name in questions}
            latency_ms = float(payload.get("latency_ms") or ((time.monotonic() - started) * 1000))
            trace_id = str(payload.get("trace_id") or trace_id)
            batch = DecisionBatch(ok=True, answers=answers, latency_ms=latency_ms, trace_id=trace_id, source="laya")
            try:
                record_monitor_state(_LAST_KEY, {
                    "purpose": purpose, "latency_ms": round(latency_ms, 2),
                    "answers": {k: {"value": v.value, "confidence": round(v.confidence, 4)} for k, v in answers.items()},
                    "at": utc_now(),
                })
            except Exception:
                pass
            self._capture({
                "kind": "prediction", "trace_id": trace_id, "purpose": purpose,
                "state": state if self.capture_text else {"keys": list(state)},
                "answers": {k: {"value": v.value, "confidence": v.confidence, "probabilities": v.probabilities} for k, v in answers.items()},
                "at": utc_now(),
            })
            return batch
        except Exception as exc:
            return DecisionBatch(ok=False, latency_ms=(time.monotonic() - started) * 1000, error=str(exc), trace_id=trace_id)

    def route_turn(self, request: str, *, previous_user: str = "", previous_frame: dict[str, Any] | None = None) -> DecisionBatch:
        if not self.routing_enabled:
            return DecisionBatch(ok=False, error="routing disabled")
        return self.predict({
            "request": request,
            "previous_user": previous_user,
            "previous_intent": str((previous_frame or {}).get("intent") or ""),
            "previous_entity": str((previous_frame or {}).get("entity") or ""),
        }, route_questions(), purpose="turn_route")

    def validate_loop(
        self,
        request: str,
        transcript: str,
        *,
        signal: dict[str, Any] | None = None,
        candidate_tools: list[str] | None = None,
        stage: str = "stall",
    ) -> dict[str, Any] | None:
        if not self.validator_enabled:
            return None
        batch = self.predict({
            "request": request,
            "stage": stage,
            "signal": signal or {},
            "candidate_tools": (candidate_tools or [])[:16],
            "recent_tool_loop": str(transcript or "")[-1200:],
        }, validator_questions(), purpose=f"validator_{stage}")
        if not batch.ok:
            return None
        action = batch.accepted("action", self.validator_threshold)
        # A corrective decision still falls through to the 2B validator, which
        # can choose an exact recovery tool. Only terminal decisions bypass it.
        if not action or action.value not in {"finish", "blocked"}:
            return None
        diagnosis_answer = batch.answers.get("diagnosis") or ChoiceDecision(value="unknown")
        diagnosis = diagnosis_answer.value or ("task_complete" if action.value == "finish" else "unknown")
        return {
            "decision": action.value,
            "suggested_tool": "",
            "diagnosis": diagnosis,
            "laya_confidence": action.confidence,
            "laya_latency_ms": batch.latency_ms,
            "laya_trace_id": batch.trace_id,
            "validator": "laya",
        }

    def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "enabled": False}
        try:
            response = requests.get(self.endpoint + "/health", timeout=min(self.timeout, 0.5))
            if response.status_code == 200:
                return dict(response.json())
            return {"ok": False, "status_code": response.status_code}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def preload_async(self) -> None:
        """Compatibility no-op: the dedicated sidecar owns asynchronous preload."""
        return None

    def preload(self, timeout: float = 0.5) -> bool:
        if not self.enabled:
            return False
        try:
            return requests.post(self.endpoint + "/preload", timeout=max(timeout, self.timeout)).status_code == 200
        except Exception:
            return False

    def unload(self, timeout: float = 1.0) -> bool:
        if not self.enabled:
            return False
        try:
            return requests.post(self.endpoint + "/unload", timeout=max(timeout, self.timeout)).status_code == 200
        except Exception:
            return False

    def _capture(self, payload: dict[str, Any]) -> None:
        if not self.capture_enabled:
            return
        try:
            path = Path(self.capture_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except OSError:
                    pass
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
                handle.flush()
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        except Exception:
            pass

    def record_outcome(
        self,
        *,
        trace_id: str,
        request: str,
        continuation: bool,
        successful_tools: list[dict[str, Any]],
        completed: bool,
        blocked: bool,
    ) -> None:
        if not trace_id or not self.capture_enabled:
            return
        names = [str(item.get("tool") or "") for item in successful_tools if item.get("success")]
        families = [tool_family_for_name(name) for name in names if name]
        family = next((x for x in families if x and x != "other"), "conversation" if not names else "other")
        mutating = any(not bool(item.get("readonly", True)) for item in successful_tools if item.get("success"))
        labels: dict[str, Any] = {
            "tool_requirement": "required" if names else "none",
            "continuation": "continuation" if continuation else "new",
            "risk": "mutation" if mutating else "read_only",
            "turn_status": "blocked" if blocked else ("complete" if completed else "unknown"),
            "tools": names,
        }
        # Successful tool execution is a strong family label. A no-tool turn is
        # not automatically "conversation" (it may be coding/explanation), so
        # omit route_family rather than poisoning future fine-tuning data.
        if names and family:
            labels["route_family"] = family
        self._capture({
            "kind": "supervision", "trace_id": trace_id,
            "request": request if self.capture_text else "",
            "labels": labels,
            "at": utc_now(),
        })


def tool_family_for_name(name: str) -> str:
    name = str(name or "")
    if name in _TOOL_TO_FAMILY:
        return _TOOL_TO_FAMILY[name]
    if name.startswith(("read_", "file_", "path_", "json_", "csv_", "text_", "regex_", "yaml_")):
        return "files"
    if name.startswith(("network_", "dns_", "tcp_", "tls_", "http_", "route_", "neighbor_", "socket_", "scan_")):
        return "network"
    if name.startswith(("web_", "page_", "fetch_", "extract_")):
        return "web"
    if name.startswith(("repo_", "search_repo", "get_repo")):
        return "coding"
    if name.startswith(("host_", "process_", "service_", "pressure_", "filesystem_")):
        return "system"
    if "memory" in name or name in {"remember", "profile_image_info"}:
        return "memory"
    if "recipe" in name or "pipeline" in name:
        return "workflow"
    if "research" in name:
        return "research"
    if "reminder" in name or name in {"queue_work", "notify_desktop"}:
        return "automation"
    return "other"


def recommended_tools_for_family(family: str) -> tuple[str, ...]:
    return _FAMILY_TOOLS.get(str(family or ""), ())
