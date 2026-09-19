# Refactor notes

This refactor intentionally preserves public entry points and tool names while moving implementation behind focused modules.

Key compatibility facades:
- `agent.py` -> `al_agent/*`
- `worker.py` -> `al_agent/background/*`
- `tools/__init__.py` -> `tools/catalog.py` + memory exports
- `tools/primitive_ops.py` -> `tools/primitive_modules/*`

Extension seams are documented in `ARCHITECTURE.md`.  Existing recipes remain valid because registered public tool names did not change.
