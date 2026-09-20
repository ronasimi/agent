#!/usr/bin/env bash
set -euo pipefail

create_alias() {
  local alias="$1" source="$2"
  local mf
  mf="$(mktemp)"
  trap 'rm -f "$mf"' RETURN
  cat >"$mf" <<EOF
FROM ${source}
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
EOF
  ollama create "$alias" -f "$mf"
  rm -f "$mf"
  trap - RETURN
}

create_alias agent-report:9b 'hf.co/empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M'
create_alias agent-main:4b   'hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q8_0'
create_alias agent-fast:2b   'hf.co/empero-ai/Qwen3.8-2B-Distill-GGUF:Q4_K_M'

printf '\nCreated aliases:\n'
ollama list | grep -E '^(agent-report|agent-main|agent-fast):' || true
