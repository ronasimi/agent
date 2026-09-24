#!/usr/bin/env python3
"""Run an optional BrowserGym task through an external policy callable.

Example:
  python diagnostics/benchmarks/benchmark_browsergym.py --probe
  python diagnostics/benchmarks/benchmark_browsergym.py --env browsergym/miniwob.click-test \
      --policy my_policy:decide

The policy signature is ``decide(observation, budget_snapshot)`` and may return
an action dict/string or ``(action, metrics)`` where metrics contains prompt /
output token counts and an optional recovery flag.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.browser_benchmark import BenchmarkBudget, BrowserGymRunner, browsergym_available


def load_policy(spec: str):
    if ":" not in spec:
        raise ValueError("--policy must be module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--env", default="")
    parser.add_argument("--policy", default="")
    parser.add_argument("--task-kwargs", default="{}", help="JSON object forwarded to gym.make task_kwargs")
    parser.add_argument("--benchmark", default="browsergym")
    args = parser.parse_args()

    if args.probe or not args.env:
        print(json.dumps(browsergym_available(), indent=2, sort_keys=True))
        return 0
    if not args.policy:
        parser.error("--policy module:function is required when --env is supplied")
    task_kwargs = json.loads(args.task_kwargs)
    if not isinstance(task_kwargs, dict):
        raise ValueError("--task-kwargs must decode to an object")
    result = BrowserGymRunner().run(
        args.env,
        load_policy(args.policy),
        task_kwargs=task_kwargs,
        budget=BenchmarkBudget.from_env(),
        benchmark=args.benchmark,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
