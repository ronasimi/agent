#!/usr/bin/env bash
set -euo pipefail

# Tiny decision/validator role: constrained routing, validation and plan arbitration.
decision_src="qwen2.5-coder:0.5b"
# Default executor/tool-caller: routine native tools, bounded arguments and prose.
executor_src="qwen2.5-coder:1.5b"
# 4B text reasoning role: explicit Think, complex analysis, bounded escalation.
reasoning_src="hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M"
# Official multimodal 4B runner. The reasoning GGUF above does not include a
# vision projector, so vision stays on this separate model rather than silently
# pretending the text-only alias can inspect pixels.
vision_src="qwen3.5:4b"
# 9B report/research role: explicit long-form research/report synthesis only.
report_src="qwen3.5:9b"

ollama pull "$decision_src"
ollama cp "$decision_src" agent-micro

ollama pull "$executor_src"
ollama cp "$executor_src" agent-main

ollama pull "$reasoning_src"
ollama cp "$reasoning_src" agent-reasoning

ollama pull "$vision_src"

ollama pull "$report_src"
ollama cp "$report_src" agent-research

printf '\nConfigured model roles:\n'
ollama show agent-micro --modelfile
ollama show agent-main --modelfile
ollama show agent-reasoning --modelfile
ollama show "$vision_src" --modelfile
ollama show agent-research --modelfile
