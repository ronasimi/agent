#!/usr/bin/env python3
"""Deterministic, model-free size gate for optimization candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json"}
EXCLUDED_PARTS = {".git", "workspace", "memory", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-source-tokens", type=int, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    files = []
    total_bytes = 0
    total_lines = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or set(path.relative_to(root).parts) & EXCLUDED_PARTS:
            continue
        if path.suffix.lower() not in SUFFIXES and path.name != "Dockerfile":
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:
            continue
        files.append(path.relative_to(root).as_posix())
        total_bytes += len(raw)
        total_lines += raw.count(b"\n") + bool(raw)
    estimated_tokens = (total_bytes + 3) // 4
    result = {
        "files": len(files),
        "bytes": total_bytes,
        "lines": total_lines,
        "estimated_tokens": estimated_tokens,
        "max_source_tokens": args.max_source_tokens,
        "passed": estimated_tokens <= args.max_source_tokens,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
