"""Configuration loading with container and source-tree defaults."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def config_path() -> Path:
    explicit = os.environ.get("AGENT_CONFIG")
    if explicit:
        return Path(explicit)
    container = Path("/app/config/config.yaml")
    if container.exists():
        return container
    return Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config() -> dict:
    path = config_path()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return normalize_config(raw)


DEFAULT_MODEL = "agent-main:4b"


def normalize_config(raw: dict) -> dict:
    """Migrate legacy role settings onto one authoritative model and context.

    Role keys are compatibility aliases for installed tools/extensions only.
    They never select another runner, even when an old config still contains
    executor/fast/report/vision overrides.
    """
    import copy
    import warnings

    if not isinstance(raw, dict):
        raise TypeError("Configuration must be a mapping")
    config = copy.deepcopy(raw)
    agent = config.setdefault("agent", {})
    if not isinstance(agent, dict):
        raise TypeError("agent configuration must be a mapping")
    model = str(
        os.environ.get("AGENT_MODEL") or agent.get("model") or DEFAULT_MODEL
    ).strip()
    if not model:
        raise ValueError("agent.model must be nonempty")
    options = dict(
        agent.get("main_options")
        or {"num_ctx": 32768, "temperature": 0.2, "num_predict": 2048}
    )
    options.setdefault("num_ctx", 32768)
    if int(options["num_ctx"]) < 2048:
        raise ValueError("agent.main_options.num_ctx must be at least 2048")
    agent.update(model=model, main_options=options)
    # Legacy model-based router settings are intentionally ignored. Tool routing
    # is now a deterministic catalog prefilter feeding the same resident model.
    legacy_router = agent.pop("router", None)
    if legacy_router:
        warnings.warn(
            "Legacy agent.router settings are ignored; tool routing is deterministic "
            "and uses the resident main model.",
            stacklevel=2,
        )
    routing = dict(agent.get("tool_routing") or {})
    routing.setdefault("candidate_limit", 8)
    routing.setdefault("auto_activate_threshold", 0.80)
    routing.setdefault("auto_activate_margin", 0.20)
    routing.setdefault("min_candidate_score", 0.18)
    if not 1 <= int(routing["candidate_limit"]) <= 8:
        raise ValueError("agent.tool_routing.candidate_limit must be between 1 and 8")
    for key in ("auto_activate_threshold", "auto_activate_margin", "min_candidate_score"):
        value = float(routing[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"agent.tool_routing.{key} must be between 0 and 1")
        routing[key] = value
    agent["tool_routing"] = routing
    requested_protocol = str(agent.get("tool_protocol") or "qwen_xml").strip().lower()
    if requested_protocol not in {"qwen_xml", "native"}:
        warnings.warn(
            f"The configured Qwen3.8 model does not use the legacy {requested_protocol!r} tool protocol; "
            "using qwen_xml instead.",
            stacklevel=2,
        )
        requested_protocol = "qwen_xml"
    agent["tool_protocol"] = requested_protocol
    agent.setdefault("thinking_default", False)
    # The configured Qwen3.8 template exposes Ollama's thinking toggle.  Keeping
    # this enabled is what makes ``think: false`` reach ``enable_thinking=false``
    # and therefore emit the template's empty <think></think> generation prefix.
    agent["supports_thinking"] = True
    keep_alive = agent.get("keep_alive", -1)
    roles = (
        "executor",
        "decision",
        "reasoning",
        "fast",
        "vision",
        "report",
        "compaction",
    )
    ignored = [
        role for role in roles if agent.get(role + "_model") not in (None, "", model)
    ]
    if ignored:
        warnings.warn(
            "Single-model mode ignores legacy role overrides: " + ", ".join(ignored),
            stacklevel=2,
        )
    for role in roles:
        agent[role + "_model"] = model
        agent[role + "_options"] = dict(options)
        agent[role + "_model_keep_alive"] = keep_alive
    agent["context"] = {
        **dict(agent.get("context") or {}),
        "num_ctx": options["num_ctx"],
    }
    agent["semantic_memory_enabled"] = False
    agent["report_restore_models_after_stage"] = False
    agent["model_escalation"] = {"enabled": False}
    agent["structured_plan"] = {"enabled": False}
    agent["tool_loop_validator"] = {"enabled": False}
    agent["model_capabilities"] = {
        **dict(agent.get("model_capabilities") or {}),
        "enabled": False,
    }
    agent["warmup"] = {
        **dict(agent.get("warmup") or {}),
        "fast_model_prewarm": False,
        "decision_model_prewarm": False,
    }
    agent["warmup"].setdefault("prime_tool_schemas", True)
    agent["host"] = os.environ.get("OLLAMA_HOST") or agent.get(
        "host", "http://127.0.0.1:11434"
    )
    return config
