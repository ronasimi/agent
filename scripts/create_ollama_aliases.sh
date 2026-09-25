#!/usr/bin/env bash
set -euo pipefail
# One original distilled 4B model. AGENT_MODEL_SOURCE can select an existing tag.
agent_model_source="${AGENT_MODEL_SOURCE:-hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M}"
agent_model_alias="${AGENT_MODEL:-agent-main:4b}"
ollama pull "$agent_model_source"
if [[ "$agent_model_source" != "$agent_model_alias" ]]; then
  ollama cp "$agent_model_source" "$agent_model_alias"
fi
ollama show "$agent_model_alias" --modelfile
