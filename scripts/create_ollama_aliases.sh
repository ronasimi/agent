#!/usr/bin/env bash
set -euo pipefail

# Stable role aliases for the locally pulled base Qwen3.5 models.
# `ollama cp` reuses the existing blobs instead of duplicating model storage.
for alias in agent-main:4b agent-fast:2b agent-report:9b; do
  ollama rm "$alias" >/dev/null 2>&1 || true
done

ollama cp qwen3.5:4b agent-main:4b
ollama cp qwen3.5:2b agent-fast:2b
ollama cp qwen3.5:9b agent-report:9b

printf '\nCreated aliases:\n'
ollama list | grep -E '^(agent-report|agent-main|agent-fast):' || true
