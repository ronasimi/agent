"""Manage optional workspace custom tools."""
from __future__ import annotations

import os
import subprocess
import tempfile

TOOLS_DIR = "/app/workspace/custom_tools"
os.makedirs(TOOLS_DIR, exist_ok=True)


def list_tool_files() -> str:
    """List optional custom tool source files in the workspace tool directory."""
    try:
        files = sorted(f for f in os.listdir(TOOLS_DIR) if f.endswith(".py"))
        return f"Custom tool files in {TOOLS_DIR}:\n" + ("\n".join(f"- {f}" for f in files) or "(none)")
    except Exception as exc:
        return f"Error: {exc}"


def create_or_update_tool(filename: str, python_code: str) -> str:
    """Syntax-check and save a custom tool; custom functions must use @agent_tool to become callable."""
    filename = os.path.basename(str(filename))
    if not filename.endswith(".py") or filename.startswith("_"):
        return "Error: Filename must end with .py and may not start with '_'."
    code = str(python_code)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name
    try:
        result = subprocess.run(["python", "-m", "py_compile", temp_path], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return f"Syntax validation failed; tool was not saved.\n{result.stderr or result.stdout}"
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    path = os.path.join(TOOLS_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)
    except Exception as exc:
        return f"Error saving tool: {exc}"
    return f"Saved custom tool to {path}. Call /reload to activate decorated @agent_tool functions."


def read_tool_source(filename: str) -> str:
    """Read a custom tool source file."""
    filename = os.path.basename(str(filename))
    path = os.path.join(TOOLS_DIR, filename if filename.endswith(".py") else filename + ".py")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except Exception as exc:
        return f"Error: {exc}"


def reload_tools() -> str:
    """Reload the explicit builtin registry and workspace custom tools."""
    from . import load_tools, AVAILABLE_TOOLS_MAP
    count, errors = load_tools()
    text = f"Tools reloaded: {count} active tools.\nActive: {', '.join(AVAILABLE_TOOLS_MAP.keys())}"
    if errors:
        text += "\n\nLoad errors:\n" + "\n".join(f"- {name}: {error}" for name, error in errors.items())
    return text
