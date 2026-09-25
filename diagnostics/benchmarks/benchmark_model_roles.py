#!/usr/bin/env python3
"""Benchmark configured Ollama roles with explicit cold/warm separation.

The benchmark mirrors the harness' effective options, reports every latency
sample, captures runner residency, and measures executor foreground TTFT while the decision
model is being restored in the background.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def _maybe_reexec_in_repo_venv() -> None:
    if os.environ.get("AGENT_VENV_REEXEC") == "1" or not VENV_PYTHON.is_file():
        return
    try:
        current = Path(sys.executable).resolve()
        venv_python = VENV_PYTHON.resolve()
    except OSError:
        return
    if current == venv_python:
        return
    env = dict(os.environ)
    env["AGENT_VENV_REEXEC"] = "1"
    os.execve(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


_maybe_reexec_in_repo_venv()

try:
    import yaml
    from ollama import Client
except ModuleNotFoundError as exc:
    missing = exc.name or "required Python dependency"
    raise SystemExit(
        f"Missing Python dependency: {missing}\n"
        f"Bootstrap the repository environment first:\n  {ROOT / 'scripts' / 'bootstrap_venv.sh'}\n"
        f"Then rerun this command. The script automatically uses {VENV_PYTHON}."
    ) from None


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _message_field(obj: Any, key: str) -> str:
    message = _value(obj, "message", {}) or {}
    return str(_value(message, key, "") or "")


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "runs": len(values),
        "median_ms": round(statistics.median(values), 2) if values else None,
        "p95_ms": round(_p95(values), 2) if values else None,
        "min_ms": round(min(values), 2) if values else None,
        "max_ms": round(max(values), 2) if values else None,
        "samples_ms": [round(value, 2) for value in values],
    }


def _timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    started = time.monotonic()
    result = fn()
    return (time.monotonic() - started) * 1000.0, result


def _unload(client: Client, model: str) -> None:
    try:
        client.chat(model=model, messages=[], keep_alive=0, stream=False, think=False)
    except Exception:
        pass


def _unload_all(client: Client, models: list[str]) -> None:
    for model in dict.fromkeys(model for model in models if model):
        _unload(client, model)


def _warm_chat(client: Client, model: str, options: dict[str, Any], keep_alive: Any) -> float:
    elapsed, _ = _timed(lambda: client.chat(
        model=model,
        messages=[],
        options=dict(options or {}),
        keep_alive=keep_alive,
        stream=False,
        think=False,
    ))
    return elapsed


def _main_ttft_once(client: Client, model: str, options: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    started = time.monotonic()
    first_model: float | None = None
    stream = None
    try:
        stream = client.chat(
            model=model,
            messages=[{"role": "user", "content": "In one sentence, explain what a TCP socket is."}],
            options={**options, "num_predict": min(64, int(options.get("num_predict", 64) or 64))},
            keep_alive=-1,
            stream=True,
            think=False,
        )
        for chunk in stream:
            elapsed = (time.monotonic() - started) * 1000.0
            if first_model is None and (_message_field(chunk, "thinking") or _message_field(chunk, "content")):
                first_model = elapsed
            if _message_field(chunk, "content").strip():
                return first_model or elapsed, elapsed, None
        return first_model, None, "stream ended before visible content"
    except Exception as exc:
        return first_model, None, f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def _main_latency(client: Client, model: str, options: dict[str, Any], runs: int) -> dict[str, Any]:
    _unload(client, model)
    cold_model, cold_visible, cold_error = _main_ttft_once(client, model, options)
    warmup_ms: float | None = None
    warmup_error: str | None = None
    try:
        warmup_ms = _warm_chat(client, model, options, -1)
    except Exception as exc:
        warmup_error = f"{type(exc).__name__}: {exc}"

    visible: list[float] = []
    model_tokens: list[float] = []
    errors: list[str] = []
    for _ in range(runs):
        first_model, first_visible, error = _main_ttft_once(client, model, options)
        if first_model is not None:
            model_tokens.append(first_model)
        if first_visible is not None:
            visible.append(first_visible)
        if error:
            errors.append(error)
    warm = _summary(visible)
    return {
        **warm,
        "cold": {
            "first_model_token_ms": round(cold_model, 2) if cold_model is not None else None,
            "first_visible_token_ms": round(cold_visible, 2) if cold_visible is not None else None,
            "error": cold_error,
        },
        "warm": warm,
        "warm_first_model_token": _summary(model_tokens),
        "explicit_warmup_ms": round(warmup_ms, 2) if warmup_ms is not None else None,
        "warmup_error": warmup_error,
        "thinking_enabled": False,
        "failures": len(errors),
        "errors": sorted(set(errors))[:5],
    }


def _validator_once(client: Client, model: str, options: dict[str, Any], keep_alive: Any) -> tuple[float | None, str | None]:
    fmt = {
        "type": "object",
        "properties": {"decision": {"type": "string", "enum": ["finish", "recover", "blocked"]}},
        "required": ["decision"],
        "additionalProperties": False,
    }
    prompt = (
        "User asked: check whether api.example.com resolves. "
        "Observed: DNS returned SERVFAIL and no endpoint probe has run. "
        "Return whether the task is finished, needs recovery, or is blocked."
    )
    started = time.monotonic()
    try:
        client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={**options, "num_predict": min(48, int(options.get("num_predict", 48) or 48))},
            keep_alive=keep_alive,
            format=fmt,
            stream=False,
            think=False,
        )
        return (time.monotonic() - started) * 1000.0, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _fast_validator(client: Client, model: str, options: dict[str, Any], keep_alive: Any, runs: int) -> dict[str, Any]:
    _unload(client, model)
    cold_value, cold_error = _validator_once(client, model, options, keep_alive)
    warmup_ms: float | None = None
    warmup_error: str | None = None
    try:
        warmup_ms = _warm_chat(client, model, options, keep_alive)
    except Exception as exc:
        warmup_error = f"{type(exc).__name__}: {exc}"

    values: list[float] = []
    errors: list[str] = []
    for _ in range(runs):
        value, error = _validator_once(client, model, options, keep_alive)
        if value is not None:
            values.append(value)
        if error:
            errors.append(error)
    warm = _summary(values)
    return {
        **warm,
        "cold": {"latency_ms": round(cold_value, 2) if cold_value is not None else None, "error": cold_error},
        "warm": warm,
        "explicit_warmup_ms": round(warmup_ms, 2) if warmup_ms is not None else None,
        "warmup_error": warmup_error,
        "thinking_enabled": False,
        "failures": len(errors),
        "errors": sorted(set(errors))[:5],
    }


def _report_throughput(client: Client, model: str, options: dict[str, Any], keep_alive: Any, runs: int) -> dict[str, Any]:
    latencies: list[float] = []
    tok_s: list[float] = []
    failures: list[str] = []
    prompt = (
        "Write a compact technical note explaining why deterministic grounding gates improve "
        "reliability in tool-using local language-model agents. Use about 180 words."
    )
    for _ in range(runs):
        started = time.monotonic()
        try:
            result = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={**options, "num_predict": min(256, int(options.get("num_predict", 256) or 256))},
                keep_alive=keep_alive,
                stream=False,
                think=False,
            )
            latencies.append((time.monotonic() - started) * 1000.0)
            count = float(_value(result, "eval_count", 0) or 0)
            duration_ns = float(_value(result, "eval_duration", 0) or 0)
            if count > 0 and duration_ns > 0:
                tok_s.append(count / (duration_ns / 1_000_000_000.0))
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    out = _summary(latencies)
    out.update({
        "eval_tokens_per_second_samples": [round(value, 2) for value in tok_s],
        "median_eval_tokens_per_second": round(statistics.median(tok_s), 2) if tok_s else None,
        "thinking_enabled": False,
        "failures": len(failures),
        "errors": sorted(set(failures))[:5],
    })
    return out


def _installed_models(client: Client) -> set[str]:
    try:
        response = client.list()
        models = _value(response, "models", []) or []
        return {str(_value(item, "model", _value(item, "name", "")) or "") for item in models}
    except Exception:
        return set()


def _embedding_latency(client: Client, model: str, runs: int, *, enabled: bool) -> dict[str, Any]:
    installed = model in _installed_models(client)
    if not enabled or not installed:
        note = "semantic memory is disabled" if not enabled else "embedding model is not installed"
        return {
            **_summary([]), "enabled": enabled, "installed": installed, "failures": 0, "errors": [],
            "note": note, "remediation": None if installed else f"ollama pull {model}",
        }
    values: list[float] = []
    failures: list[str] = []
    for _ in range(runs):
        started = time.monotonic()
        try:
            client.embed(model=model, input=["semantic retrieval benchmark for local agent memory"], keep_alive="2m")
            values.append((time.monotonic() - started) * 1000.0)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    return {
        **_summary(values), "enabled": enabled, "installed": installed,
        "failures": len(failures), "errors": sorted(set(failures))[:5],
    }


def _residency_snapshot(client: Client) -> dict[str, Any]:
    try:
        response = client.ps()
        rows = []
        for item in _value(response, "models", []) or []:
            size = int(_value(item, "size", 0) or 0)
            vram = int(_value(item, "size_vram", 0) or 0)
            rows.append({
                "name": str(_value(item, "name", "") or ""),
                "model": str(_value(item, "model", _value(item, "name", "")) or ""),
                "context_length": int(_value(item, "context_length", 0) or 0),
                "size_bytes": size,
                "size_vram_bytes": vram,
                "expires_at": str(_value(item, "expires_at", "") or ""),
                "vram_percent": round((vram / size) * 100.0, 1) if size else None,
            })
        return {"available": True, "models": rows}
    except Exception as exc:
        return {"available": False, "models": [], "error": f"{type(exc).__name__}: {exc}"}


def _foreground_during_decision_prewarm(
    client: Client,
    host: str,
    executor_model: str,
    executor_options: dict[str, Any],
    decision_model: str,
    decision_options: dict[str, Any],
    decision_keep_alive: Any,
    delay_ms: float,
) -> dict[str, Any]:
    """Measure executor TTFT while another connection restores the decision model."""
    _warm_chat(client, executor_model, executor_options, -1)
    _unload(client, decision_model)
    _, baseline, baseline_error = _main_ttft_once(client, executor_model, executor_options)
    started = threading.Event()
    result: dict[str, Any] = {"prewarm_ms": None, "prewarm_error": None}

    def _prewarm() -> None:
        background_client = Client(host=host, timeout=180)
        started.set()
        try:
            result["prewarm_ms"] = round(_warm_chat(background_client, decision_model, decision_options, decision_keep_alive), 2)
        except Exception as exc:
            result["prewarm_error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_prewarm, name="benchmark-decision-prewarm", daemon=True)
    thread.start()
    started.wait(timeout=5.0)
    time.sleep(max(0.0, delay_ms) / 1000.0)
    _, foreground, foreground_error = _main_ttft_once(client, executor_model, executor_options)
    thread.join(timeout=190.0)
    added = foreground - baseline if foreground is not None and baseline is not None else None
    ratio = foreground / baseline if foreground is not None and baseline else None
    return {
        "prewarm_start_delay_ms": round(max(0.0, delay_ms), 2),
        "baseline_executor_ttft_ms": round(baseline, 2) if baseline is not None else None,
        "foreground_executor_ttft_ms": round(foreground, 2) if foreground is not None else None,
        "added_latency_ms": round(added, 2) if added is not None else None,
        "slowdown_ratio": round(ratio, 2) if ratio is not None else None,
        "baseline_error": baseline_error,
        "foreground_error": foreground_error,
        "prewarm_ms": result["prewarm_ms"],
        "prewarm_error": result["prewarm_error"],
        "prewarm_finished": not thread.is_alive(),
    }


def _model_residency(
    client: Client,
    host: str,
    roles: list[tuple[str, str, dict[str, Any], Any]],
    contention_delay_ms: float,
) -> dict[str, Any]:
    by_role = {role: (model, options, keep_alive) for role, model, options, keep_alive in roles}
    all_models = [model for _, model, _, _ in roles]
    cold: dict[str, Any] = {}
    for role, model, options, keep_alive in roles:
        _unload_all(client, all_models)
        try:
            cold[role] = round(_warm_chat(client, model, options, keep_alive), 2)
        except Exception as exc:
            cold[role] = {"error": f"{type(exc).__name__}: {exc}"}

    executor_model, executor_options, executor_keep_alive = by_role["executor"]
    decision_model, decision_options, decision_keep_alive = by_role["decision"]
    report_model, report_options, report_keep_alive = by_role["report"]
    transitions: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}

    _unload_all(client, all_models)
    _warm_chat(client, executor_model, executor_options, executor_keep_alive)
    snapshots["executor_only"] = _residency_snapshot(client)
    transitions["executor->decision_cold_beside_executor"] = round(
        _warm_chat(client, decision_model, decision_options, decision_keep_alive), 2
    )
    snapshots["executor_decision_after_decision_load"] = _residency_snapshot(client)
    transitions["warm_decision->executor"] = round(
        _warm_chat(client, executor_model, executor_options, executor_keep_alive), 2
    )
    snapshots["after_warm_executor_switch"] = _residency_snapshot(client)
    transitions["warm_executor->decision"] = round(
        _warm_chat(client, decision_model, decision_options, decision_keep_alive), 2
    )
    snapshots["after_warm_decision_switch"] = _residency_snapshot(client)

    if "reasoning" in by_role:
        reasoning_model, reasoning_options, reasoning_keep_alive = by_role["reasoning"]
        _unload(client, decision_model)
        transitions["executor+decision->reasoning_evict_decision"] = round(
            _warm_chat(client, reasoning_model, reasoning_options, reasoning_keep_alive), 2
        )
        snapshots["executor_reasoning"] = _residency_snapshot(client)
        _unload(client, reasoning_model)
        transitions["reasoning->decision_restore"] = round(
            _warm_chat(client, decision_model, decision_options, decision_keep_alive), 2
        )
        snapshots["executor_decision_restored"] = _residency_snapshot(client)

    _unload(client, executor_model)
    _unload(client, decision_model)
    transitions["interactive->report"] = round(_warm_chat(client, report_model, report_options, report_keep_alive), 2)
    snapshots["report_only"] = _residency_snapshot(client)
    _unload(client, report_model)
    transitions["report->executor_restore"] = round(
        _warm_chat(client, executor_model, executor_options, executor_keep_alive), 2
    )
    snapshots["after_report_executor_restore"] = _residency_snapshot(client)
    transitions["executor->decision_background_restore_after_report"] = round(
        _warm_chat(client, decision_model, decision_options, decision_keep_alive), 2
    )
    snapshots["after_background_decision_restore"] = _residency_snapshot(client)
    transitions["warm_decision->executor_after_report"] = round(
        _warm_chat(client, executor_model, executor_options, executor_keep_alive), 2
    )

    contention = _foreground_during_decision_prewarm(
        client, host, executor_model, executor_options, decision_model, decision_options,
        decision_keep_alive, contention_delay_ms,
    )
    snapshots["after_foreground_contention_test"] = _residency_snapshot(client)
    return {
        "cold_load_ms": cold,
        "transition_ms": transitions,
        "foreground_during_background_decision_prewarm": contention,
        "residency_snapshots": snapshots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20, help="warm executor/decision/embed repetitions")
    parser.add_argument("--report-runs", type=int, default=1, help="9B report repetitions")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--contention-delay-ms", type=float, default=250.0,
                        help="delay after starting decision prewarm before the executor foreground probe")
    parser.add_argument("--skip-residency", "--skip-load-swap", dest="skip_residency", action="store_true",
                        help="skip explicit unload/load, residency, and contention measurements")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    agent = dict(cfg.get("agent") or {})
    host = str(agent.get("host") or "http://127.0.0.1:11434")
    client = Client(host=host, timeout=180)
    executor_model = str(agent.get("executor_model") or agent.get("model") or "agent-main")
    decision_model = str(agent.get("decision_model") or "agent-micro")
    reasoning_model = str(agent.get("reasoning_model") or "agent-reasoning")
    report_model = str(agent.get("report_model") or "agent-research")
    embed_model = str(agent.get("embed_model") or "nomic-embed-text")
    main_options = dict(agent.get("main_options") or {})
    decision_options = dict(agent.get("decision_options") or {})
    reasoning_options = dict(agent.get("reasoning_options") or {})
    fast_options = decision_options  # compatibility output alias only
    validator_cfg = dict(agent.get("tool_loop_validator") or {})
    validator_options = {**decision_options, **dict(validator_cfg.get("options") or {})}
    report_options = dict(agent.get("report_options") or {})
    decision_keep_alive = agent.get("decision_model_keep_alive", -1)
    validator_keep_alive = validator_cfg.get("keep_alive", decision_keep_alive)
    reasoning_keep_alive = agent.get("reasoning_model_keep_alive", "2m")
    report_keep_alive = agent.get("report_model_keep_alive", "10m")
    semantic_enabled = bool(agent.get("semantic_memory_enabled", False))
    warmup_cfg = dict(agent.get("warmup") or {})
    runs = max(1, int(args.runs))
    report_runs = max(1, int(args.report_runs))

    output: dict[str, Any] = {
        "host": host,
        "roles": {
            "executor": executor_model,
            "decision": decision_model,
            "reasoning": reasoning_model,
            "report": report_model,
            "embedding": embed_model,
        },
        "effective_config": {
            "executor_num_ctx": main_options.get("num_ctx"),
            "decision_num_ctx": decision_options.get("num_ctx"),
            "reasoning_num_ctx": reasoning_options.get("num_ctx"),
            "main_num_ctx": main_options.get("num_ctx"),
            "fast_num_ctx": fast_options.get("num_ctx"),
            "validator_num_ctx": validator_options.get("num_ctx"),
            "report_num_ctx": report_options.get("num_ctx"),
            "report_decision_restore_mode": "nonblocking_prewarm",
            "decision_keep_alive": decision_keep_alive,
            "startup_decision_prewarm": bool(warmup_cfg.get("decision_model_prewarm", True)),
            # Compatibility diagnostics for older parsers.
            "fast_keep_alive": decision_keep_alive,
            "startup_fast_prewarm": bool(warmup_cfg.get("fast_model_prewarm", True)),
            "thinking_default": bool(agent.get("thinking_default", False)),
            "semantic_memory_enabled": semantic_enabled,
            "warm_samples_per_role": runs,
        },
        "executor_ttft": _main_latency(client, executor_model, main_options, runs),
        "decision_validator": _fast_validator(client, decision_model, validator_options, validator_keep_alive, runs),
        "reasoning_ttft": _main_latency(client, reasoning_model, reasoning_options, runs),
        "embedding_latency": _embedding_latency(client, embed_model, runs, enabled=semantic_enabled),
    }

    # Compatibility keys retained for existing tooling.
    output["main_ttft"] = output["executor_ttft"]
    output["fast_validator"] = output["decision_validator"]

    output["report_throughput"] = _report_throughput(
        client, report_model, report_options, report_keep_alive, report_runs
    )
    if not args.skip_residency:
        output["model_residency"] = _model_residency(client, host, [
            ("executor", executor_model, main_options, -1),
            ("decision", decision_model, decision_options, decision_keep_alive),
            ("reasoning", reasoning_model, reasoning_options, reasoning_keep_alive),
            ("report", report_model, report_options, report_keep_alive),
        ], max(0.0, float(args.contention_delay_ms)))

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
