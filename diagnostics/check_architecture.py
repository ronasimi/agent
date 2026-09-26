#!/usr/bin/env python3
"""Fail fast if key modular architecture invariants regress."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
limits = {
    "al_agent/runtime.py": 8_000,
    "worker.py": 4_000,
    "tools/primitive_ops.py": 5_000,
}
errors=[]
for removed in ("agent.py", "al_agent/cli.py", "al_agent/cli_commands.py"):
    if (ROOT / removed).exists(): errors.append(f"removed CLI surface returned: {removed}")
for rel, limit in limits.items():
    size=(ROOT/rel).stat().st_size
    if size>limit: errors.append(f"{rel} grew to {size} bytes (limit {limit})")

sys.path.insert(0,str(ROOT))
try:
    from tools.providers import BUILTINS
    if len(BUILTINS)!=len(set(BUILTINS)): errors.append("duplicate builtin tool provider specs")
    from al_agent.background.handlers import JOB_HANDLERS
    if len(JOB_HANDLERS)!=len(set(JOB_HANDLERS)): errors.append("duplicate background handler names")
    from diagnostics.check_turing_completeness import check_contract
    errors.extend(check_contract())
except Exception as exc:
    errors.append(f"provider discovery failed: {exc}")

if errors:
    print("Architecture check failed:")
    for error in errors: print(" -",error)
    raise SystemExit(1)
print("Architecture check passed")
