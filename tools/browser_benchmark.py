"""Browser benchmark budgets, BrowserGym compatibility, and regression metrics.

BrowserGym is intentionally optional: the normal local harness does not need the
benchmark packages installed.  When available, :class:`BrowserGymRunner` adapts
our compact browser-action dictionaries to BrowserGym's Gymnasium reset/step
contract and high-level action strings.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .browser_state import BrowserStateStore, _connect


@dataclass
class BenchmarkBudget:
    """Hard per-task resource ceilings used by browser/UI benchmarks."""

    max_model_calls: int = 40
    max_browser_actions: int = 50
    max_wall_time_s: float = 300.0
    max_prompt_tokens: int = 120_000
    max_output_tokens: int = 30_000
    max_recovery_attempts: int = 8

    model_calls: int = 0
    browser_actions: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    recovery_attempts: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_env(cls) -> "BenchmarkBudget":
        def _i(name: str, default: int) -> int:
            try:
                return max(1, int(os.getenv(name, str(default))))
            except (TypeError, ValueError):
                return default

        def _f(name: str, default: float) -> float:
            try:
                return max(1.0, float(os.getenv(name, str(default))))
            except (TypeError, ValueError):
                return default

        return cls(
            max_model_calls=_i("AGENT_UI_BUDGET_MODEL_CALLS", 40),
            max_browser_actions=_i("AGENT_UI_BUDGET_BROWSER_ACTIONS", 50),
            max_wall_time_s=_f("AGENT_UI_BUDGET_WALL_SECONDS", 300.0),
            max_prompt_tokens=_i("AGENT_UI_BUDGET_PROMPT_TOKENS", 120_000),
            max_output_tokens=_i("AGENT_UI_BUDGET_OUTPUT_TOKENS", 30_000),
            max_recovery_attempts=_i("AGENT_UI_BUDGET_RECOVERIES", 8),
        )

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def violation(self) -> str | None:
        checks = (
            (self.model_calls > self.max_model_calls, "model_calls"),
            (self.browser_actions > self.max_browser_actions, "browser_actions"),
            (self.elapsed_s > self.max_wall_time_s, "wall_time"),
            (self.prompt_tokens > self.max_prompt_tokens, "prompt_tokens"),
            (self.output_tokens > self.max_output_tokens, "output_tokens"),
            (self.recovery_attempts > self.max_recovery_attempts, "recovery_attempts"),
        )
        return next((name for failed, name in checks if failed), None)

    def consume_model(self, *, prompt_tokens: int = 0, output_tokens: int = 0) -> None:
        self.model_calls += 1
        self.prompt_tokens += max(0, int(prompt_tokens or 0))
        self.output_tokens += max(0, int(output_tokens or 0))

    def consume_browser(self, *, recovery: bool = False) -> None:
        self.browser_actions += 1
        if recovery:
            self.recovery_attempts += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "elapsed_s": round(self.elapsed_s, 3),
            "violation": self.violation(),
        }


class BrowserBenchmarkStore:
    """Persist benchmark results in the harness SQLite database."""

    def __init__(self) -> None:
        self._ensure_schema()

    @staticmethod
    def _ensure_schema() -> None:
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS browser_benchmark_runs (
                    id TEXT PRIMARY KEY,
                    benchmark TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    success INTEGER NOT NULL DEFAULT 0,
                    reward REAL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    budget_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_browser_benchmark_created
                  ON browser_benchmark_runs(created_at DESC);
                """
            )

    def record_run(
        self,
        *,
        benchmark: str,
        task_id: str,
        success: bool,
        reward: float | None,
        metrics: dict[str, Any],
        budget: dict[str, Any],
        error: str = "",
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or uuid.uuid4().hex
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO browser_benchmark_runs(
                    id, benchmark, task_id, success, reward, metrics_json,
                    budget_json, error, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    str(benchmark or "browser")[:160],
                    str(task_id or "")[:300],
                    1 if success else 0,
                    None if reward is None else float(reward),
                    json.dumps(metrics or {}, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(budget or {}, ensure_ascii=False, separators=(",", ":")),
                    str(error or "")[:4000],
                    created,
                ),
            )
        return run_id

    def recent_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM browser_benchmark_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                metrics = json.loads(row["metrics_json"] or "{}")
            except Exception:
                metrics = {}
            try:
                budget = json.loads(row["budget_json"] or "{}")
            except Exception:
                budget = {}
            result.append({
                "id": row["id"],
                "benchmark": row["benchmark"],
                "task_id": row["task_id"],
                "success": bool(row["success"]),
                "reward": row["reward"],
                "metrics": metrics,
                "budget": budget,
                "error": row["error"],
                "created_at": row["created_at"],
            })
        return result


def _metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get("metrics", {}).get(key, default)
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return round(values[index], 3)


def _aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "runs": 0, "success_rate": None, "avg_steps": None,
            "p50_task_ms": None, "p95_task_ms": None,
            "avg_prompt_tokens": None, "avg_output_tokens": None,
        }
    task_ms = [_metric(row, "task_ms") for row in rows if _metric(row, "task_ms") > 0]
    steps = [_metric(row, "steps") for row in rows]
    prompts = [_metric(row, "prompt_tokens") for row in rows]
    outputs = [_metric(row, "output_tokens") for row in rows]
    return {
        "runs": len(rows),
        "success_rate": round(sum(1 for row in rows if row.get("success")) / len(rows), 4),
        "avg_steps": round(statistics.fmean(steps), 3) if steps else None,
        "p50_task_ms": _pct(task_ms, 0.50),
        "p95_task_ms": _pct(task_ms, 0.95),
        "avg_prompt_tokens": round(statistics.fmean(prompts), 1) if prompts else None,
        "avg_output_tokens": round(statistics.fmean(outputs), 1) if outputs else None,
    }


def regression_dashboard(limit: int = 100) -> dict[str, Any]:
    """Return browser benchmark + live trajectory regression statistics."""
    BrowserStateStore._ensure_schema()
    store = BrowserBenchmarkStore()
    runs = store.recent_runs(limit=limit)
    window = min(10, max(1, len(runs) // 2 if len(runs) > 1 else 1))
    current = runs[:window]
    previous = runs[window:window * 2]

    with _connect() as conn:
        trajectory = conn.execute(
            """
            SELECT COUNT(*) AS steps,
                   SUM(CASE WHEN process_success = 1 THEN 1 ELSE 0 END) AS process_ok,
                   SUM(CASE WHEN outcome_success = 1 THEN 1 ELSE 0 END) AS outcome_ok,
                   SUM(CASE WHEN outcome_success IS NOT NULL THEN 1 ELSE 0 END) AS outcome_checks
            FROM browser_ui_trajectory
            """
        ).fetchone()

    live_steps = int((trajectory["steps"] if trajectory else 0) or 0)
    process_ok = int((trajectory["process_ok"] if trajectory else 0) or 0)
    outcome_checks = int((trajectory["outcome_checks"] if trajectory else 0) or 0)
    outcome_ok = int((trajectory["outcome_ok"] if trajectory else 0) or 0)
    return {
        "current": _aggregate_runs(current),
        "previous": _aggregate_runs(previous),
        "live_ui": {
            "steps": live_steps,
            "process_success_rate": round(process_ok / live_steps, 4) if live_steps else None,
            "outcome_checks": outcome_checks,
            "outcome_success_rate": round(outcome_ok / outcome_checks, 4) if outcome_checks else None,
        },
        "recent_runs": runs[:30],
        "window_size": window,
    }


def browsergym_available() -> dict[str, Any]:
    """Probe optional BrowserGym/Gymnasium dependencies without importing at startup."""
    status = {"available": False, "gymnasium": False, "browsergym": False, "error": ""}
    try:
        import gymnasium  # noqa: F401
        status["gymnasium"] = True
    except Exception as exc:
        status["error"] = f"gymnasium: {exc}"
        return status
    try:
        import browsergym.core  # noqa: F401
        status["browsergym"] = True
        status["available"] = True
    except Exception as exc:
        status["error"] = f"browsergym: {exc}"
    return status


class BrowserGymAdapter:
    """Translate our compact action shape to BrowserGym high-level actions."""

    @staticmethod
    def normalize_observation(obs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obs, dict):
            return {"raw": obs}
        result = {
            "goal": obs.get("goal", ""),
            "goal_object": obs.get("goal_object", ()),
            "url": obs.get("url", ""),
            "open_pages_urls": list(obs.get("open_pages_urls", ()) or ()),
            "open_pages_titles": list(obs.get("open_pages_titles", ()) or ()),
            "active_page_index": obs.get("active_page_index", 0),
            "last_action": obs.get("last_action", ""),
            "last_action_error": obs.get("last_action_error", ""),
            "elapsed_time": obs.get("elapsed_time", 0),
        }
        # BrowserGym already provides a marked accessibility tree/DOM. Keep the
        # structured objects rather than flattening them into bloated HTML.
        if "axtree_object" in obs:
            result["axtree_object"] = obs.get("axtree_object")
        if "dom_object" in obs:
            result["dom_object"] = obs.get("dom_object")
        if "extra_element_properties" in obs:
            result["extra_element_properties"] = obs.get("extra_element_properties")
        return result

    @staticmethod
    def to_browsergym_actions(action: dict[str, Any] | str) -> list[str]:
        """Translate one compact action into current BrowserGym calls.

        Most compact actions map 1:1. Closing a non-active tab requires a
        focus followed by ``tab_close()`` because BrowserGym's current
        ``tab_close`` primitive takes no index. The runner executes those as
        separate environment steps so this also works with action sets that
        disable BrowserGym multiaction parsing.
        """
        if isinstance(action, str):
            return [action]
        op = str(action.get("op") or "").lower()
        ref = str(action.get("ref") or "")
        value = str(action.get("value") or "")
        if op == "click":
            return [f"click({json.dumps(ref)})"]
        if op == "type":
            return [f"fill({json.dumps(ref)}, {json.dumps(value)})"]
        if op == "select":
            return [f"select_option({json.dumps(ref)}, {json.dumps(value)})"]
        if op == "key":
            # BrowserGym distinguishes element-scoped press(bid, key) from a
            # page-global keyboard_press(key). Match the harness semantics.
            if ref:
                return [f"press({json.dumps(ref)}, {json.dumps(value)})"]
            return [f"keyboard_press({json.dumps(value)})"]
        if op == "scroll":
            direction = str(action.get("direction") or "down").lower()
            amount = max(1, abs(int(action.get("amount", 700) or 700)))
            dx = dy = 0
            if direction == "up":
                dy = -amount
            elif direction == "left":
                dx = -amount
            elif direction == "right":
                dx = amount
            else:
                dy = amount
            return [f"scroll({dx}, {dy})"]
        if op == "navigate":
            return [f"goto({json.dumps(str(action.get('url') or ''))})"]
        if op == "back":
            return ["go_back()"]
        if op == "switch_tab":
            return [f"tab_focus({int(action.get('tab_index', 0))})"]
        if op == "close_tab":
            index = int(action.get("tab_index", -1))
            return ["tab_close()"] if index < 0 else [f"tab_focus({index})", "tab_close()"]
        if op == "new_tab":
            return ["new_tab()"]
        if op in {"done", "terminate"}:
            return [f"send_msg_to_user({json.dumps(value or 'done')})"]
        raise ValueError(f"Unsupported BrowserGym action op: {op!r}")

    @staticmethod
    def to_browsergym_action(action: dict[str, Any] | str) -> str:
        """Backward-compatible single-string representation.

        BrowserGym's default action set accepts newline-separated multiactions,
        while :class:`BrowserGymRunner` uses ``to_browsergym_actions`` and steps
        each primitive separately for compatibility with strict benchmarks.
        """
        return "\n".join(BrowserGymAdapter.to_browsergym_actions(action))


class BrowserGymRunner:
    """Small optional runner for BrowserGym-compatible regression tasks."""

    def __init__(self, *, store: BrowserBenchmarkStore | None = None) -> None:
        self.store = store or BrowserBenchmarkStore()

    def run(
        self,
        env_id: str,
        policy: Callable[[dict[str, Any], dict[str, Any]], Any],
        *,
        task_kwargs: dict[str, Any] | None = None,
        budget: BenchmarkBudget | None = None,
        benchmark: str = "browsergym",
    ) -> dict[str, Any]:
        status = browsergym_available()
        if not status["available"]:
            raise RuntimeError("BrowserGym benchmark dependencies are unavailable: " + str(status.get("error") or "unknown"))
        import importlib
        import gymnasium as gym
        import browsergym.core  # noqa: F401  # registers openended
        # Benchmark task packages register Gymnasium environments on import.
        # Keep imports lazy so the low-spec normal runtime pays no benchmark cost.
        lowered = str(env_id or "").lower()
        module_map = {
            "miniwob": "browsergym.miniwob",
            "webarena_verified": "browsergym.webarena_verified",
            "webarena": "browsergym.webarena",
            "visualwebarena": "browsergym.visualwebarena",
            "workarena": "browsergym.workarena",
            "assistantbench": "browsergym.assistantbench",
            "timewarp": "browsergym.timewarp",
        }
        for marker, module_name in module_map.items():
            if marker in lowered:
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    pass
                break

        budget = budget or BenchmarkBudget.from_env()
        started = time.perf_counter()
        env = gym.make(env_id, task_kwargs=task_kwargs) if task_kwargs else gym.make(env_id)
        reward_total = 0.0
        terminated = truncated = False
        error = ""
        success = False
        steps = 0
        try:
            obs, info = env.reset()
            while not (terminated or truncated):
                violation = budget.violation()
                if violation:
                    error = f"budget_exceeded:{violation}"
                    break
                normalized = BrowserGymAdapter.normalize_observation(obs)
                decision = policy(normalized, budget.snapshot())
                policy_metrics: dict[str, Any] = {}
                if isinstance(decision, tuple) and len(decision) == 2:
                    decision, policy_metrics = decision
                budget.consume_model(
                    prompt_tokens=int(policy_metrics.get("prompt_tokens", 0) or 0),
                    output_tokens=int(policy_metrics.get("output_tokens", 0) or 0),
                )
                violation = budget.violation()
                if violation:
                    error = f"budget_exceeded:{violation}"
                    break
                actions = BrowserGymAdapter.to_browsergym_actions(decision)
                for action in actions:
                    budget.consume_browser(recovery=bool(policy_metrics.get("recovery")))
                    violation = budget.violation()
                    if violation:
                        error = f"budget_exceeded:{violation}"
                        break
                    obs, reward, terminated, truncated, info = env.step(action)
                    reward_total += float(reward or 0.0)
                    steps += 1
                    if terminated or truncated:
                        break
                if error:
                    break
            success = bool(terminated and not error and (reward_total > 0 or bool((info or {}).get("success", False))))
        except Exception as exc:
            error = str(exc)[:2000]
        finally:
            try:
                env.close()
            except Exception:
                pass
        metrics = {
            "task_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "steps": steps,
            "model_calls": budget.model_calls,
            "browser_actions": budget.browser_actions,
            "prompt_tokens": budget.prompt_tokens,
            "output_tokens": budget.output_tokens,
            "recovery_attempts": budget.recovery_attempts,
        }
        run_id = self.store.record_run(
            benchmark=benchmark,
            task_id=env_id,
            success=success,
            reward=reward_total,
            metrics=metrics,
            budget=budget.snapshot(),
            error=error,
        )
        return {
            "run_id": run_id,
            "benchmark": benchmark,
            "task_id": env_id,
            "success": success,
            "reward": reward_total,
            "metrics": metrics,
            "budget": budget.snapshot(),
            "error": error,
        }
