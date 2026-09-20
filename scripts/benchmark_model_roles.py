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


def _message_content(obj: Any) -> str:
    message = _value(obj, "message", {}) or {}
    return str(_value(message, "content", "") or "")


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


def _main_ttft(client: Client, model: str, options: dict[str, Any], runs: int) -> dict[str, Any]:
    values: list[float] = []
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
                stream=True,
            )
            got_visible = False
            for chunk in stream:
                if _message_content(chunk).strip():
                    values.append((time.monotonic() - started) * 1000.0)
                    got_visible = True
                    break
            if not got_visible:
                failures.append("stream ended before visible content")
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    return {**_summary(values), "failures": len(failures), "errors": sorted(set(failures))[:5]}


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


def _embedding_latency(client: Client, model: str, runs: int) -> dict[str, Any]:
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
            failures.append(f"{type(exc).__name__}: {exc}")
    return {**_summary(values), "failures": len(failures), "errors": sorted(set(failures))[:5]}


def _load_and_swap(client: Client, roles: list[tuple[str, str, dict[str, Any], Any]]) -> dict[str, Any]:
    cold_loads: dict[str, Any] = {}
    for role, model, options, keep_alive in roles:
        _unload(client, model)
        try:
            cold_loads[role] = round(_warm_chat(client, model, options, keep_alive), 2)
        except Exception as exc:
            cold_loads[role] = {"error": f"{type(exc).__name__}: {exc}"}

    swaps: dict[str, Any] = {}
    for (from_role, from_model, _, _), (to_role, to_model, to_options, to_keep_alive) in zip(roles, roles[1:]):
        _unload(client, from_model)
        try:
            swaps[f"{from_role}->{to_role}"] = round(_warm_chat(client, to_model, to_options, to_keep_alive), 2)
        except Exception as exc:
            swaps[f"{from_role}->{to_role}"] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"cold_load_ms": cold_loads, "swap_warm_ms": swaps}


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
    report_options = dict(agent.get("report_options") or {})
    fast_keep_alive = agent.get("fast_model_keep_alive", "2m")
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
        "main_ttft": _main_ttft(client, main_model, main_options, runs),
        "fast_validator": _fast_validator(client, fast_model, fast_options, fast_keep_alive, runs),
        "embedding_latency": _embedding_latency(client, embed_model, runs),
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
