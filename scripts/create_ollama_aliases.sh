#!/usr/bin/env bash
set -euo pipefail
# Main alias defaults to the runtime config. Existing local aliases are reused;
# otherwise provide AGENT_MODEL_SOURCE explicitly so this script never guesses
# a 4B upstream tag.
agent_model_alias="${AGENT_MODEL:-agent-main:4b}"
agent_model_source="${AGENT_MODEL_SOURCE:-}"
if ! ollama show "$agent_model_alias" >/dev/null 2>&1; then
  if [[ -z "$agent_model_source" ]]; then
    echo "Missing $agent_model_alias. Set AGENT_MODEL_SOURCE to the desired 4B source tag." >&2
    exit 2
  fi
  ollama pull "$agent_model_source"
  if [[ "$agent_model_source" != "$agent_model_alias" ]]; then
    ollama cp "$agent_model_source" "$agent_model_alias"
  fi
fi
ollama show "$agent_model_alias" --modelfile
# Small stateless System-1 router used for tool selection.
router_model="${AGENT_ROUTER_MODEL:-qwen2.5:0.5b}"
ollama pull "$router_model"
