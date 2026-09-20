#!/usr/bin/env python3
"""Join Laya prediction/supervision trace rows into a compact local dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/app/memory/laya_training.jsonl")
    parser.add_argument("--output", default="/app/workspace/laya_harness_training.jsonl")
    args = parser.parse_args()
    predictions = {}
    supervision = {}
    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"No capture file at {source}")
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        trace = str(row.get("trace_id") or "")
        if not trace:
            continue
        if row.get("kind") == "prediction" and row.get("purpose") == "turn_route":
            predictions[trace] = row
        elif row.get("kind") == "supervision":
            supervision[trace] = row
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for trace, pred in predictions.items():
            sup = supervision.get(trace)
            if not sup:
                continue
            state = pred.get("state") or {}
            row = {
                "trace_id": trace,
                "request": str((state or {}).get("request") or sup.get("request") or ""),
                "previous_user": str((state or {}).get("previous_user") or ""),
                "predicted": pred.get("answers") or {},
                "labels": sup.get("labels") or {},
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"ok": True, "examples": count, "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
