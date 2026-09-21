#!/usr/bin/env bash
set -euo pipefail

service="${AGENT_SOAK_SERVICE:-worker}"

if ! docker compose ps --services --status running | grep -Fxq "$service"; then
  echo "Soak-test service '$service' is not running." >&2
  echo "Start the stack first with: docker compose up -d --build" >&2
  exit 2
fi

exec docker compose exec -w /app "$service" \
  python scripts/soak_test_tools.py \
  --workers 2 \
  --mutating-mode isolated \
  "$@"
