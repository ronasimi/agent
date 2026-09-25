"""Cached Ollama model capability/conformance probes.

The harness supports model aliases that may be rebuilt in-place.  A small set of
safe startup probes records how the *actual configured runner* behaves instead
of assuming every chat template supports Ollama's optional fields identically.

Probing is deliberately best-effort and background-friendly:
- profiles are cached by model identity (digest/template hash + probe version),
- inconclusive probes remain ``None`` rather than disabling a feature,
- only explicit provider/template rejection marks a request feature unsupported,
- no user data or tools with side effects are involved.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .model_protocol import extract_qwen_xml_tool_calls, is_prompt_protocol_error, tool_result_message

PROBE_VERSION = 2
DEFAULT_CACHE_PATH = "/app/memory/model_capabilities.json"
_CACHE_LOCK = threading.RLock()
_ACTIVE: dict[str, "ModelCapabilityProfile"] = {}
_UNSET = object()


class ModelCapabilityError(RuntimeError):
    """Raised when a cached/probed model explicitly rejects a required feature."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _message(response: Any) -> Any:
    return _field(response, "message", {}) or {}


def _message_content(response: Any) -> str:
    return str(_field(_message(response), "content", "") or "")


def _message_thinking(response: Any) -> str:
    return str(_field(_message(response), "thinking", "") or "")


def _message_tool_calls(response: Any) -> list[Any]:
    raw = _field(_message(response), "tool_calls", []) or []
    if isinstance(raw, dict):
        return [raw]
    try:
        return list(raw)
    except TypeError:
        return [raw]


def _message_dict(response: Any) -> dict[str, Any]:
    message = _message(response)
    if isinstance(message, dict):
        return {k: v for k, v in message.items() if k in {"role", "content", "thinking", "images", "tool_name", "tool_calls"}}
    result: dict[str, Any] = {}
    for key in ("role", "content", "thinking", "images", "tool_name", "tool_calls"):
        value = getattr(message, key, None)
        if value not in (None, "", [], {}):
            result[key] = value
    result.setdefault("role", "assistant")
    return result


def _iter_stream(response: Any):
    if isinstance(response, dict) or hasattr(response, "message"):
        yield response
        return
    yield from response


def _unsupported_feature(exc: Exception, feature: str) -> bool:
    text = str(exc or "").lower()
    feature = str(feature or "").lower()
    generic = ("unsupported", "not support", "unknown field", "unknown parameter", "invalid option")
    if not any(marker in text for marker in generic):
        return False
    aliases = {
        "think": ("think", "thinking"),
        "tools": ("tool", "function calling", "function_call"),
    }.get(feature, (feature,))
    return any(alias in text for alias in aliases)


@dataclass(frozen=True)
class ModelCapabilityProfile:
    model: str
    identity: str
    probe_version: int = PROBE_VERSION
    probed_at: str = field(default_factory=_utc_now)
    plain_chat: bool | None = None
    content_streaming: bool | None = None
    think_parameter: bool | None = None
    reasoning_streaming: bool | None = None
    tools_parameter: bool | None = None
    tool_call_mode: str = "unknown"  # native | qwen_xml | accepted_unverified | unsupported | unknown
    tool_result_continuation: bool | None = None
    metadata_capabilities: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata_capabilities"] = list(self.metadata_capabilities)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCapabilityProfile":
        values = dict(data or {})
        values["metadata_capabilities"] = tuple(values.get("metadata_capabilities") or ())
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})


def get_active_model_capabilities(model: str) -> ModelCapabilityProfile | None:
    with _CACHE_LOCK:
        return _ACTIVE.get(str(model or ""))


def publish_model_capabilities(profile: ModelCapabilityProfile) -> ModelCapabilityProfile:
    with _CACHE_LOCK:
        _ACTIVE[profile.model] = profile
    return profile


def _model_rows(client: Any) -> list[Any]:
    try:
        response = client.list()
    except Exception:
        return []
    rows = _field(response, "models", []) or []
    try:
        return list(rows)
    except TypeError:
        return []


def _model_name(row: Any) -> str:
    return str(_field(row, "model", None) or _field(row, "name", None) or "")


def resolve_model_identity(client: Any, model: str) -> tuple[str, tuple[str, ...]]:
    """Return a stable cache identity and Ollama-advertised capability hints."""
    model = str(model or "").strip()
    digest = ""
    for row in _model_rows(client):
        if _model_name(row) == model:
            digest = str(_field(row, "digest", "") or "")
            if digest:
                break

    template = ""
    capabilities: tuple[str, ...] = ()
    try:
        shown = client.show(model)
        template = str(_field(shown, "template", "") or "")
        raw_caps = _field(shown, "capabilities", []) or []
        capabilities = tuple(sorted(str(item) for item in raw_caps))
    except Exception:
        shown = None

    server_version = ""
    try:
        version_fn = getattr(client, "version", None)
        if callable(version_fn):
            version_response = version_fn()
            server_version = str(_field(version_response, "version", "") or "")
    except Exception:
        server_version = ""

    if digest:
        token = digest
    elif template:
        token = "template-" + hashlib.sha256(template.encode("utf-8", errors="replace")).hexdigest()
    else:
        token = "name-" + hashlib.sha256(model.encode("utf-8", errors="replace")).hexdigest()
    version_token = hashlib.sha256(server_version.encode("utf-8")).hexdigest()[:12] if server_version else "unknown-server"
    return f"v{PROBE_VERSION}:{model}:{token}:{version_token}", capabilities


def _read_cache(path: str) -> dict[str, Any]:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {"version": PROBE_VERSION, "profiles": {}}
    if int(raw.get("version") or 0) != PROBE_VERSION or not isinstance(raw.get("profiles"), dict):
        return {"version": PROBE_VERSION, "profiles": {}}
    return raw


def _write_cache(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


def load_cached_model_capabilities(client: Any, model: str, *, cache_path: str = DEFAULT_CACHE_PATH) -> ModelCapabilityProfile | None:
    identity, _caps = resolve_model_identity(client, model)
    with _CACHE_LOCK:
        raw = _read_cache(cache_path)
        data = (raw.get("profiles") or {}).get(identity)
        if not isinstance(data, dict):
            return None
        try:
            profile = ModelCapabilityProfile.from_dict(data)
        except (TypeError, ValueError):
            return None
        return publish_model_capabilities(profile)


def _save_profile(profile: ModelCapabilityProfile, cache_path: str) -> None:
    with _CACHE_LOCK:
        raw = _read_cache(cache_path)
        profiles = dict(raw.get("profiles") or {})
        profiles[profile.identity] = profile.to_dict()
        # Bound stale aliases/rebuilds without needing a separate cleanup job.
        if len(profiles) > 32:
            ordered = sorted(
                profiles.items(),
                key=lambda item: str((item[1] or {}).get("probed_at") or ""),
                reverse=True,
            )[:32]
            profiles = dict(ordered)
        _write_cache(cache_path, {"version": PROBE_VERSION, "profiles": profiles})


def _probe_stream_chat(
    client: Any, model: str, options: dict[str, Any], keep_alive: Any, *, think: Any = _UNSET
) -> tuple[bool, bool, bool, bool]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "options": {**options, "num_predict": min(12, max(8, int(options.get("num_predict") or 12)))},
        "keep_alive": keep_alive,
        "stream": True,
    }
    if think is not _UNSET:
        kwargs["think"] = think
    response = client.chat(**kwargs)
    streaming_transport = not (isinstance(response, dict) or hasattr(response, "message"))
    saw_chunk = False
    saw_content = False
    saw_thinking = False
    for chunk in _iter_stream(response):
        saw_chunk = True
        saw_content = saw_content or bool(_message_content(chunk))
        saw_thinking = saw_thinking or bool(_message_thinking(chunk))
    return saw_chunk, saw_content, saw_thinking, streaming_transport


def _probe_tools(client: Any, model: str, options: dict[str, Any], keep_alive: Any, *, think_supported: bool | None) -> tuple[bool | None, str, bool | None]:
    schema = {
        "type": "function",
        "function": {
            "name": "capability_probe_echo",
            "description": "Return the provided text. This is a harmless startup conformance probe.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Call capability_probe_echo exactly once with text=PING. Do not answer in prose."}],
        "tools": [schema],
        "options": {**options, "num_predict": min(32, max(16, int(options.get("num_predict") or 32)))},
        "keep_alive": keep_alive,
        "stream": False,
    }
    if think_supported is not False:
        kwargs["think"] = False
    try:
        response = client.chat(**kwargs)
    except Exception as exc:
        if _unsupported_feature(exc, "tools"):
            return False, "unsupported", None
        return None, "unknown", None

    calls = _message_tool_calls(response)
    content = _message_content(response)
    xml_calls, _xml_errors = extract_qwen_xml_tool_calls(content)
    if calls:
        mode = "native"
    elif xml_calls:
        mode = "qwen_xml"
    else:
        return True, "accepted_unverified", None

    # Verify that an assistant tool invocation can be followed by a native tool
    # result without the model/template rejecting the transaction.
    assistant = _message_dict(response)
    continuation_messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Call capability_probe_echo exactly once with text=PING. Do not answer in prose."},
        assistant,
        tool_result_message("capability_probe_echo", "PONG"),
    ]
    cont_kwargs: dict[str, Any] = {
        "model": model,
        "messages": continuation_messages,
        "options": {**options, "num_predict": min(12, max(8, int(options.get("num_predict") or 12)))},
        "keep_alive": keep_alive,
        "stream": False,
    }
    if think_supported is not False:
        cont_kwargs["think"] = False
    try:
        follow = client.chat(**cont_kwargs)
    except Exception as exc:
        if is_prompt_protocol_error(exc) or _unsupported_feature(exc, "tools"):
            return True, mode, False
        return True, mode, None
    if _message_content(follow) or _message_thinking(follow) or _message_tool_calls(follow):
        return True, mode, True
    return True, mode, None


def probe_model_capabilities(
    client: Any,
    model: str,
    *,
    options: dict[str, Any] | None = None,
    keep_alive: Any = -1,
    cache_path: str = DEFAULT_CACHE_PATH,
    force: bool = False,
) -> ModelCapabilityProfile:
    """Probe one model with bounded, harmless chat/tool requests and cache it."""
    model = str(model or "").strip()
    base_options = dict(options or {})
    base_options["temperature"] = 0.0
    identity, metadata_caps = resolve_model_identity(client, model)

    if not force:
        with _CACHE_LOCK:
            raw = _read_cache(cache_path)
            cached = (raw.get("profiles") or {}).get(identity)
            if isinstance(cached, dict):
                try:
                    return publish_model_capabilities(ModelCapabilityProfile.from_dict(cached))
                except (TypeError, ValueError):
                    pass

    plain_chat: bool | None = None
    content_streaming: bool | None = None
    think_parameter: bool | None = None
    reasoning_streaming: bool | None = None
    errors: list[str] = []

    # First test literal think=false because that is the harness's normal fast
    # path. If the model/template explicitly rejects it, retry plain chat without
    # the optional field and remember to omit ``think`` at runtime.
    try:
        saw, content, first_thinking, streamed = _probe_stream_chat(client, model, base_options, keep_alive, think=False)
        plain_chat = bool(saw and (content or first_thinking))
        content_streaming = bool(streamed and content)
        think_parameter = True
    except Exception as exc:
        if _unsupported_feature(exc, "think"):
            think_parameter = False
        else:
            errors.append(f"think=false/plain chat: {exc}")
        try:
            saw, content, fallback_thinking, streamed = _probe_stream_chat(client, model, base_options, keep_alive)
            plain_chat = bool(saw and (content or fallback_thinking))
            content_streaming = bool(streamed and content)
        except Exception as fallback_exc:
            plain_chat = False
            errors.append(f"plain chat: {fallback_exc}")

    if think_parameter is True:
        try:
            _saw, _content, thinking, streamed = _probe_stream_chat(client, model, base_options, keep_alive, think=True)
            reasoning_streaming = bool(streamed and thinking)
        except Exception as exc:
            if _unsupported_feature(exc, "think"):
                # Some templates accept false as a compatibility no-op but reject
                # enabling reasoning. Keep the parameter usable for nothink while
                # advertising that streamed reasoning is unavailable.
                reasoning_streaming = False
            else:
                errors.append(f"think=true: {exc}")

    tools_parameter, tool_call_mode, tool_result_continuation = _probe_tools(
        client, model, base_options, keep_alive, think_supported=think_parameter
    )

    profile = ModelCapabilityProfile(
        model=model,
        identity=identity,
        plain_chat=plain_chat,
        content_streaming=content_streaming,
        think_parameter=think_parameter,
        reasoning_streaming=reasoning_streaming,
        tools_parameter=tools_parameter,
        tool_call_mode=tool_call_mode,
        tool_result_continuation=tool_result_continuation,
        metadata_capabilities=metadata_caps,
        error="; ".join(errors)[:2000],
    )
    publish_model_capabilities(profile)
    # Do not make a transiently unavailable server/model sticky across restarts.
    # A valid plain-chat exchange is the minimum bar for persistent conformance.
    if profile.plain_chat is True:
        try:
            _save_profile(profile, cache_path)
        except OSError:
            pass
    return profile


def schedule_model_capability_probe(
    client: Any,
    model: str,
    *,
    options: dict[str, Any] | None = None,
    keep_alive: Any = -1,
    cache_path: str = DEFAULT_CACHE_PATH,
    force: bool = False,
    on_complete: Callable[[ModelCapabilityProfile], None] | None = None,
    busy_check: Callable[[], bool] | None = None,
    idle_delay_seconds: float = 0.0,
) -> threading.Thread:
    """Load/probe one model on a daemon thread without blocking startup.

    Cached profiles are published immediately after cheap model-identity lookup.
    A cold conformance run can optionally wait for an interactive-idle window so
    startup diagnostics never intentionally queue ahead of a user turn.
    """
    def _run() -> None:
        if not force:
            cached = load_cached_model_capabilities(client, model, cache_path=cache_path)
            if cached is not None:
                if on_complete is not None:
                    on_complete(cached)
                return

        delay = max(0.0, float(idle_delay_seconds or 0.0))
        # Require one continuous idle window rather than sleeping blindly. If a
        # turn starts during the delay, restart the window after it finishes.
        idle_started = time.monotonic()
        while True:
            busy = bool(busy_check and busy_check())
            if busy:
                idle_started = time.monotonic()
                time.sleep(0.25)
                continue
            if time.monotonic() - idle_started >= delay:
                break
            time.sleep(min(0.25, max(0.01, delay)))

        profile = probe_model_capabilities(
            client, model, options=options, keep_alive=keep_alive,
            cache_path=cache_path, force=force,
        )
        if on_complete is not None:
            on_complete(profile)

    thread = threading.Thread(target=_run, name=f"model-capabilities-{model}", daemon=True)
    thread.start()
    return thread


def capability_chat_overrides(
    model: str,
    *,
    think: Any = _UNSET,
    tools: Any = _UNSET,
) -> dict[str, Any]:
    """Return safe optional Ollama kwargs for the active model profile.

    Unknown/inconclusive profiles preserve the historical behavior. Only an
    explicit conformance rejection suppresses an optional provider field.
    """
    profile = get_active_model_capabilities(model)
    result: dict[str, Any] = {}
    from tools.config import load_config
    think_enabled = bool(load_config()["agent"].get("supports_thinking", False))
    if think_enabled and think is not _UNSET and not (profile is not None and profile.think_parameter is False):
        result["think"] = think
    if tools is not _UNSET:
        # Never serialize an explicitly empty tool list. Omitting the optional
        # field is the broadest cross-template representation of a direct-chat
        # turn and matches Ollama's minimal API examples.
        if tools:
            if profile is not None and profile.tools_parameter is False:
                raise ModelCapabilityError(
                    f"model '{model}' rejected Ollama tool schemas during startup conformance probing"
                )
            if profile is not None and profile.tool_call_mode == "accepted_unverified":
                raise ModelCapabilityError(
                    f"model '{model}' accepted tool schemas but did not emit a verifiable tool call during behavioral conformance probing"
                )
            if profile is not None and profile.tool_result_continuation is False:
                raise ModelCapabilityError(
                    f"model '{model}' rejected assistant/tool-result continuation during startup conformance probing"
                )
            result["tools"] = tools
    return result
