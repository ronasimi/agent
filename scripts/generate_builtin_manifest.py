#!/usr/bin/env python3
"""Regenerate the lightweight builtin-tool schema manifest.

The runtime catalog uses this generated file to avoid importing heavyweight tool
implementation modules at startup. Use ``--check`` in CI to detect stale schema
metadata after adding/changing a builtin tool.
"""
from __future__ import annotations

import argparse
import importlib
import pprint
from pathlib import Path

from tools.providers import BUILTINS, MUTATING_TOOLS, REPEAT_SAFE_TOOLS, SAFE_ARTIFACT_TOOLS
from tools.tool_registry import function_schema

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tools" / "builtin_manifest.py"


def build_entries() -> list[dict]:
    entries: list[dict] = []
    for module_name, function_name in BUILTINS:
        module = importlib.import_module(f"tools.{module_name}")
        func = getattr(module, function_name)
        entries.append({
            "module": module_name,
            "function": function_name,
            "schema": function_schema(func),
            "readonly": False if function_name in MUTATING_TOOLS else bool(getattr(func, "_agent_tool_readonly", True)),
            "repeat_safe": function_name in REPEAT_SAFE_TOOLS or bool(getattr(func, "_agent_tool_repeat_safe", False)),
            "safe_artifact": function_name in SAFE_ARTIFACT_TOOLS or bool(getattr(func, "_agent_tool_safe_artifact", False)),
            "timeout": getattr(func, "_agent_tool_timeout", None),
        })
    return entries


def render() -> str:
    entries = build_entries()
    return (
        '"""Generated lightweight metadata for builtin tools. Regenerate with scripts/generate_builtin_manifest.py."""\n'
        + "BUILTIN_MANIFEST = "
        + pprint.pformat(entries, width=120, sort_dicts=False)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != expected:
            print("builtin manifest is stale; run scripts/generate_builtin_manifest.py")
            return 1
        print(f"builtin manifest is current ({len(build_entries())} tools)")
        return 0
    TARGET.write_text(expected, encoding="utf-8")
    print(f"wrote {TARGET} ({len(build_entries())} tools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
