#!/usr/bin/env python3
"""Measure local Laya decision coverage/precision against captured turn outcomes.

This is deliberately an empirical deployment aid, not a claim of global
calibration. It joins turn-route prediction rows with the harness supervision
rows written after successful turns and reports accuracy at confidence gates.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_THRESHOLDS = (0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99)


def _load(path: Path):
    predictions = {}
    labels = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        trace = str(row.get("trace_id") or "")
        if not trace:
            continue
        if row.get("kind") == "prediction" and row.get("purpose") == "turn_route":
            predictions[trace] = row.get("answers") or {}
        elif row.get("kind") == "supervision":
            labels[trace] = row.get("labels") or {}
    return predictions, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/app/memory/laya_training.jsonl")
    parser.add_argument("--min-precision", type=float, default=0.97)
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"No Laya capture file at {path}")
    predictions, supervision = _load(path)
    pairs = defaultdict(list)
    for trace, answers in predictions.items():
        labels = supervision.get(trace) or {}
        for field, truth in labels.items():
            if field in {"tools", "turn_status"} or field not in answers:
                continue
            answer = answers.get(field) or {}
            predicted = str(answer.get("value") or "")
            try:
                confidence = float(answer.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if predicted and truth:
                pairs[field].append((predicted, str(truth), confidence))

    report = {"input": str(path), "joined_traces": len(set(predictions) & set(supervision)), "decisions": {}}
    for field, rows in sorted(pairs.items()):
        thresholds = []
        suggestion = None
        for threshold in DEFAULT_THRESHOLDS:
            accepted = [row for row in rows if row[2] >= threshold]
            correct = sum(1 for pred, truth, _ in accepted if pred == truth)
            precision = correct / len(accepted) if accepted else None
            coverage = len(accepted) / len(rows) if rows else 0.0
            item = {
                "threshold": threshold,
                "accepted": len(accepted),
                "precision": round(precision, 4) if precision is not None else None,
                "coverage": round(coverage, 4),
            }
            thresholds.append(item)
            if suggestion is None and precision is not None and precision >= args.min_precision and len(accepted) >= 10:
                suggestion = threshold
        report["decisions"][field] = {
            "labeled_examples": len(rows),
            "suggested_threshold": suggestion,
            "thresholds": thresholds,
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
