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
        or {"num_ctx": 16384, "temperature": 0.2, "num_predict": 2048}
    )
    options.setdefault("num_ctx", 16384)
    if int(options["num_ctx"]) < 2048:
        raise ValueError("agent.main_options.num_ctx must be at least 2048")
    agent.update(model=model, main_options=options)
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
    agent["host"] = os.environ.get("OLLAMA_HOST") or agent.get(
        "host", "http://127.0.0.1:11434"
    )
    return config
