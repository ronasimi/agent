#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: Python interpreter '$PYTHON_BIN' was not found" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required; found {sys.version.split()[0]}")
PY

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating virtual environment: $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
else
  echo "Using existing virtual environment: $VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -e "$ROOT"

cat <<MSG

Virtual environment ready.

Activate it with:
  source "$VENV/bin/activate"

Or run repository scripts directly; host-side scripts that support the bootstrap
will automatically re-exec with .venv/bin/python when the environment exists.

Examples:
  python diagnostics/benchmarks/benchmark_model_roles.py --runs 5
  python diagnostics/soak/soak_test_tools.py --duration 24h --workers 2 --mutating-mode isolated
MSG
