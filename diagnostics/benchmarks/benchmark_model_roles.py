#!/usr/bin/env python3
"""Measure the single configured model. The filename is retained for compatibility."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def _maybe_reexec_in_repo_venv():
    if (
        VENV_PYTHON.exists()
        and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve()
    ):
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])


_maybe_reexec_in_repo_venv()
sys.path.insert(0, str(ROOT))
# Run scripts/bootstrap_venv.sh first to install dependencies.
import argparse
import json
import statistics
import time

import yaml  # compatibility: dependencies are imported only after venv selection
from ollama import Client
from tools.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--report-runs", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config()["agent"]
    client = Client(
        host=config["host"],
        timeout=config.get("model_transport", {}).get("timeout_seconds", 60),
    )
    rows = []
    for _ in range(max(1, args.runs)):
        start = time.monotonic()
        first = None
        text = ""
        stream = client.chat(
            model=config["model"],
            messages=[{"role": "user", "content": "Reply with one short greeting."}],
            options=config["main_options"],
            keep_alive=config.get("keep_alive", -1),
            stream=True,
        )
        try:
            for chunk in stream:
                content = chunk.message.content or ""
                if content and first is None:
                    first = time.monotonic()
                text += content
        finally:
            stream.close()
        if not text.strip():
            raise RuntimeError("Model completed without visible content")
        rows.append(
            {
                "first_content_ms": round((first - start) * 1000, 2),
                "completion_ms": round((time.monotonic() - start) * 1000, 2),
            }
        )
    print(
        json.dumps(
            {
                "model": config["model"],
                "num_ctx": config["main_options"]["num_ctx"],
                "runs": rows,
                "median_first_content_ms": statistics.median(
                    r["first_content_ms"] for r in rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
