#!/usr/bin/env python3
"""Benchmark the configured Ollama model roles on the deployment host.

This is an observational benchmark, not a correctness test.  It measures the
latency characteristics that matter to the harness without importing runtime
state or initializing the agent database.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def _maybe_reexec_in_repo_venv() -> None:
    """Use the repo-local venv automatically when it has been bootstrapped."""
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
    os.execve(
        str(venv_python),
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


_maybe_reexec_in_repo_venv()

try:
    import yaml
    from ollama import Client
except ModuleNotFoundError as exc:
    missing = exc.name or "required Python dependency"
    raise SystemExit(
        f"Missing Python dependency: {missing}\n"
        f"Bootstrap the repository environment first:\n"
        f"  {ROOT / 'scripts' / 'bootstrap_venv.sh'}\n"
        f"Then rerun this command. The script will automatically use {VENV_PYTHON}."
    ) from None


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _message_field(obj: Any, key: str) -> Any:
    message = _value(obj, "message", {}) or {}
    return _value(message, key, None)


def _message_content(obj: Any) -> str:
    return str(_message_field(obj, "content") or "")


def _message_thinking(obj: Any) -> str:
    return str(_message_field(obj, "thinking") or "")


def _message_tool_calls(obj: Any) -> list[Any]:
    value = _message_field(obj, "tool_calls") or []
    return list(value) if isinstance(value, (list, tuple)) else []


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
    }


def _timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    started = time.monotonic()
    result = fn()
    return (time.monotonic() - started) * 1000.0, result


def _unload(client: Client, model: str) -> None:
    try:
        client.chat(model=model, messages=[], keep_alive=0, stream=False)
    except Exception:
        pass


def _warm_chat(client: Client, model: str, options: dict[str, Any], keep_alive: Any) -> float:
    elapsed, _ = _timed(lambda: client.chat(
        model=model,
        messages=[],
        options=dict(options or {}),
        keep_alive=keep_alive,
        stream=False,
    ))
    return elapsed


def _ps_snapshot(client: Client) -> dict[str, Any]:
    """Return a compact Ollama residency snapshot suitable for JSON output."""
    try:
        response = client.ps()
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "models": []}

    rows = []
    for item in list(_value(response, "models", []) or []):
        size = _value(item, "size", None)
        size_vram = _value(item, "size_vram", None)
        row = {
            "name": str(_value(item, "name", _value(item, "model", "")) or ""),
            "model": str(_value(item, "model", _value(item, "name", "")) or ""),
            "context_length": _value(item, "context_length", None),
            "size_bytes": size,
            "size_vram_bytes": size_vram,
            "expires_at": str(_value(item, "expires_at", "") or ""),
        }
        try:
            if size and size_vram is not None:
                row["vram_percent"] = round((float(size_vram) / float(size)) * 100.0, 1)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        rows.append(row)
    return {"available": True, "models": rows}


def _main_ttft(
    client: Client,
    model: str,
    options: dict[str, Any],
    runs: int,
    *,
    thinking_enabled: bool = False,
) -> dict[str, Any]:
    """Measure first model activity and first user-visible content.

    The harness passes ``think=thinking_enabled`` explicitly.  The benchmark
    must do the same: Qwen3.5 may otherwise spend the entire small token budget
    in a hidden thinking field, which looks like a failed TTFT probe even though
    the model is actively streaming.
    """
    first_token_values: list[float] = []
    first_visible_values: list[float] = []
    failures: list[str] = []
    run_options = {**options, "num_predict": min(64, int(options.get("num_predict", 64) or 64))}
    for _ in range(runs):
        started = time.monotonic()
        try:
            stream = client.chat(
                model=model,
                messages=[{"role": "user", "content": "In one sentence, explain what a TCP socket is."}],
                options=run_options,
                keep_alive=-1,
                think=thinking_enabled,
                stream=True,
            )
            first_token: float | None = None
            first_visible: float | None = None
            for chunk in stream:
                now = time.monotonic()
                content = _message_content(chunk)
                thinking = _message_thinking(chunk)
                calls = _message_tool_calls(chunk)
                if first_token is None and (content or thinking or calls):
                    first_token = (now - started) * 1000.0
                if first_visible is None and content.strip():
                    first_visible = (now - started) * 1000.0
                    break
            if first_token is not None:
                first_token_values.append(first_token)
            if first_visible is not None:
                first_visible_values.append(first_visible)
            else:
                failures.append("stream ended before visible content")
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    out = _summary(first_visible_values)
    out.update({
        "first_model_token_ms": _summary(first_token_values),
        "thinking_enabled": bool(thinking_enabled),
        "failures": len(failures),
        "errors": sorted(set(failures))[:5],
    })
    return out


def _fast_validator(client: Client, model: str, options: dict[str, Any], keep_alive: Any, runs: int) -> dict[str, Any]:
    values: list[float] = []
    failures: list[str] = []
    fmt = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["finish", "recover", "blocked"]},
        },
        "required": ["decision"],
        "additionalProperties": False,
    }
    prompt = (
        "User asked: check whether api.example.com resolves. "
        "Observed: DNS returned SERVFAIL and no endpoint probe has run. "
        "Return whether the task is finished, needs recovery, or is blocked."
    )
    run_options = {**options, "num_predict": min(48, int(options.get("num_predict", 48) or 48))}
    for _ in range(runs):
        started = time.monotonic()
        try:
            client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=run_options,
                keep_alive=keep_alive,
                format=fmt,
                think=False,
                stream=False,
            )
            values.append((time.monotonic() - started) * 1000.0)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    return {**_summary(values), "failures": len(failures), "errors": sorted(set(failures))[:5]}


def _report_throughput(client: Client, model: str, options: dict[str, Any], keep_alive: Any, runs: int) -> dict[str, Any]:
    latencies: list[float] = []
    tok_s: list[float] = []
    failures: list[str] = []
    run_options = {**options, "num_predict": min(256, int(options.get("num_predict", 256) or 256))}
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
                options=run_options,
                keep_alive=keep_alive,
                think=False,
                stream=False,
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
        "median_eval_tokens_per_second": round(statistics.median(tok_s), 2) if tok_s else None,
        "failures": len(failures),
        "errors": sorted(set(failures))[:5],
    })
    return out


def _embedding_latency(client: Client, model: str, runs: int, *, enabled: bool = True) -> dict[str, Any]:
    try:
        client.show(model)
        installed = True
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        missing = "not found" in str(exc).lower() or "404" in str(exc)
        return {
            **_summary([]),
            "enabled": bool(enabled),
            "installed": False if missing else None,
            "failures": 1 if enabled else 0,
            "errors": [message] if enabled else [],
            "note": "embedding model is not installed" if missing else "embedding model availability check failed",
            **({"remediation": f"ollama pull {model}"} if missing else {}),
        }

    if not enabled:
        return {
            **_summary([]),
            "enabled": False,
            "installed": installed,
            "failures": 0,
            "errors": [],
            "note": "semantic_memory_enabled is false; latency benchmark skipped",
        }

    values: list[float] = []
    failures: list[str] = []
    for _ in range(runs):
        started = time.monotonic()
        try:
            client.embed(
                model=model,
                input=["semantic retrieval benchmark for local agent memory"],
                keep_alive="2m",
            )
            values.append((time.monotonic() - started) * 1000.0)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(message)
            # A missing model will fail identically on every repetition. Avoid
            # wasting four more calls and return an actionable diagnostic.
            if "not found" in str(exc).lower() or "404" in str(exc):
                break
    out = {
        **_summary(values),
        "enabled": True,
        "installed": installed,
        "failures": len(failures),
        "errors": sorted(set(failures))[:5],
    }
    if failures and any("not found" in item.lower() or "404" in item for item in failures):
        out["remediation"] = f"ollama pull {model}"
    return out


def _load_and_swap(client: Client, roles: list[tuple[str, str, dict[str, Any], Any]]) -> dict[str, Any]:
    """Measure cold loads, warm role switches, and observed Ollama residency."""
    cold_loads: dict[str, Any] = {}
    role_map = {role: (model, options, keep_alive) for role, model, options, keep_alive in roles}
    all_models = list(dict.fromkeys(model for _, model, _, _ in roles if model))
    snapshots: dict[str, Any] = {}

    def unload_all() -> None:
        for model in all_models:
            _unload(client, model)

    # Pure cold-load timing: start each role from an empty role-model set so
    # another benchmarked role cannot silently change memory pressure.
    for role, model, options, keep_alive in roles:
        unload_all()
        try:
            cold_loads[role] = round(_warm_chat(client, model, options, keep_alive), 2)
        except Exception as exc:
            cold_loads[role] = {"error": f"{type(exc).__name__}: {exc}"}

    transitions: dict[str, Any] = {}
    if "main" in role_map and "fast" in role_map:
        main_model, main_options, main_keep_alive = role_map["main"]
        fast_model, fast_options, fast_keep_alive = role_map["fast"]
        unload_all()
        try:
            _warm_chat(client, main_model, main_options, main_keep_alive)
            snapshots["main_only"] = _ps_snapshot(client)
            transitions["main->fast_cold_beside_main"] = round(
                _warm_chat(client, fast_model, fast_options, fast_keep_alive), 2
            )
            snapshots["main_fast_after_fast_load"] = _ps_snapshot(client)

            # Both runners should now be resident. These two timings are the
            # actual steady-state role-switch cost rather than another load.
            transitions["warm_fast->main"] = round(
                _warm_chat(client, main_model, main_options, main_keep_alive), 2
            )
            snapshots["after_warm_main_switch"] = _ps_snapshot(client)
            transitions["warm_main->fast"] = round(
                _warm_chat(client, fast_model, fast_options, fast_keep_alive), 2
            )
            snapshots["after_warm_fast_switch"] = _ps_snapshot(client)
        except Exception as exc:
            transitions["main_fast_residency"] = {"error": f"{type(exc).__name__}: {exc}"}

    if "report" in role_map:
        report_model, report_options, report_keep_alive = role_map["report"]
        # enter_report_model_stage explicitly evicts main + fast first.
        try:
            for role in ("main", "fast"):
                if role in role_map:
                    model, _, _ = role_map[role]
                    _unload(client, model)
            transitions["interactive->report"] = round(
                _warm_chat(client, report_model, report_options, report_keep_alive), 2
            )
            snapshots["report_only"] = _ps_snapshot(client)
        except Exception as exc:
            transitions["interactive->report"] = {"error": f"{type(exc).__name__}: {exc}"}

        # Runtime report teardown now restores only main while holding the
        # inference lock. Fast is lazy-loaded later if a validator/research call
        # actually needs it, so report recovery no longer blocks a foreground
        # turn on an unnecessary 2B load.
        try:
            _unload(client, report_model)
            if "main" in role_map:
                main_model, main_options, main_keep_alive = role_map["main"]
                transitions["report->main_restore"] = round(
                    _warm_chat(client, main_model, main_options, main_keep_alive), 2
                )
                snapshots["after_report_main_restore"] = _ps_snapshot(client)
            if "fast" in role_map:
                fast_model, fast_options, fast_keep_alive = role_map["fast"]
                transitions["main->fast_lazy_after_report"] = round(
                    _warm_chat(client, fast_model, fast_options, fast_keep_alive), 2
                )
                snapshots["after_lazy_fast_restore"] = _ps_snapshot(client)
                if "main" in role_map:
                    main_model, main_options, main_keep_alive = role_map["main"]
                    transitions["warm_fast->main_after_report"] = round(
                        _warm_chat(client, main_model, main_options, main_keep_alive), 2
                    )
        except Exception as exc:
            transitions["report_recovery"] = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "cold_load_ms": cold_loads,
        "transition_ms": transitions,
        "residency_snapshots": snapshots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="interactive/fast/embed repetitions")
    parser.add_argument("--report-runs", type=int, default=1, help="9B report repetitions")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--skip-load-swap", action="store_true", help="skip explicit unload/load timing")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    agent = dict(cfg.get("agent") or {})
    host = str(agent.get("host") or "http://127.0.0.1:11434")
    client = Client(host=host, timeout=180)

    main_model = str(agent.get("model") or "agent-main:4b")
    fast_model = str(agent.get("fast_model") or "agent-fast:2b")
    report_model = str(agent.get("report_model") or "agent-report:9b")
    embed_model = str(agent.get("embed_model") or "nomic-embed-text")
    main_options = dict(agent.get("main_options") or {})
    fast_options = dict(agent.get("fast_options") or {})
    validator_cfg = dict(agent.get("tool_loop_validator") or {})
    validator_options = {**fast_options, **dict(validator_cfg.get("options") or {})}
    report_options = dict(agent.get("report_options") or {})
    fast_keep_alive = validator_cfg.get("keep_alive", agent.get("fast_model_keep_alive", "2m"))
    report_keep_alive = agent.get("report_model_keep_alive", "10m")

    runs = max(1, int(args.runs))
    report_runs = max(1, int(args.report_runs))
    output: dict[str, Any] = {
        "host": host,
        "roles": {
            "main": main_model,
            "fast": fast_model,
            "report": report_model,
            "embedding": embed_model,
        },
        "effective_config": {
            "main_num_ctx": main_options.get("num_ctx"),
            "fast_num_ctx": fast_options.get("num_ctx"),
            "validator_num_ctx": validator_options.get("num_ctx"),
            "report_num_ctx": report_options.get("num_ctx"),
            "report_fast_restore_mode": "lazy",
            "thinking_default": bool(agent.get("thinking_default", False)),
            "semantic_memory_enabled": bool(agent.get("semantic_memory_enabled", False)),
        },
        "main_ttft": _main_ttft(
            client, main_model, main_options, runs,
            thinking_enabled=bool(agent.get("thinking_default", False)),
        ),
        "fast_validator": _fast_validator(client, fast_model, validator_options, fast_keep_alive, runs),
        "embedding_latency": _embedding_latency(
            client, embed_model, runs, enabled=bool(agent.get("semantic_memory_enabled", False))
        ),
    }

    # Report synthesis intentionally runs after interactive measurements because
    # loading the 9B writer may evict smaller Ollama runners on memory-bounded hosts.
    output["report_throughput"] = _report_throughput(
        client, report_model, report_options, report_keep_alive, report_runs
    )

    if not args.skip_load_swap:
        output["model_residency"] = _load_and_swap(client, [
            ("main", main_model, main_options, -1),
            ("fast", fast_model, fast_options, fast_keep_alive),
            ("report", report_model, report_options, report_keep_alive),
        ])
        # Leave the normal interactive pair warm when possible.
        try:
            _unload(client, report_model)
            _warm_chat(client, main_model, main_options, -1)
            _warm_chat(client, fast_model, fast_options, fast_keep_alive)
        except Exception:
            pass

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
