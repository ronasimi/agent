# Refactor notes

> **Historical status:** This is a point-in-time engineering/review record and is intentionally preserved as written. Model roles, tool counts, test totals, limits, and runtime behavior may have changed since this revision. For the current harness use `CURRENT_STATE.md`, `README.md`, and `ARCHITECTURE.md`.


This refactor preserves public tool names while moving implementation behind focused modules and making the Web UI the only user interface.

Key compatibility facades:
- Web runtime -> `al_agent/runtime.py` + `webui/*`
- `worker.py` -> `al_agent/background/*`
- `tools/__init__.py` -> `tools/catalog.py` + memory exports
- `tools/primitive_ops.py` -> `tools/primitive_modules/*`

Extension seams are documented in `ARCHITECTURE.md`.  Existing recipes remain valid because registered public tool names did not change.
