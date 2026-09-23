#!/usr/bin/env bash
set -euo pipefail

main_src="hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M"
fast_src="hf.co/empero-ai/Qwen3.8-2B-Distill-GGUF:Q8_0"

ollama pull "$main_src"
ollama cp "$main_src" agent-main:4b

ollama pull "$fast_src"
ollama cp "$fast_src" agent-main:2b

printf '\nConfigured generation aliases:\n'
ollama show agent-main:4b --modelfile
ollama show agent-main:2b --modelfile
