**Autonomous Local AI Agent Harness**

A containerized, extensible CLI AI agent harness built for local LLMs via Ollama, featuring dynamic tool execution, self-healing code loops, persistent memory, and a Zsh-styled interface.

**Core Features**
* **Local LLM Integration:** Optimized for models running locally via Ollama with native tool-calling and reasoning trace support.
* **Autonomous Tool Creation:** Automatically writes, tests, and registers custom Python tools on the fly using doctests.
* **Persistent Knowledge Base:** SQLite-backed long-term memory for user preferences and facts.
* **Zsh-Styled Interactive CLI:** Enhanced prompt session with syntax highlighting, autosuggestions, persistent history, and custom styling powered by `prompt_toolkit`.
* **Containerized Security & Pentesting Suite:** Runs within an Arch Linux Docker environment equipped with standard network and security analysis utilities.

**Project Structure**
* `config/` — Configuration files and agent system prompts (`config.yaml`).
* `tools/` — Modular python tool definitions and knowledge database initialization.
* `workspace/` — Local runtime directory for scripts and generated outputs (git-ignored).
* `memory/` — Persistent volume for SQLite databases and chat histories (git-ignored).
* `agent.py` — Main entry point for the interactive agent session.
* `docker-compose.yml` — Container orchestration configuration.
