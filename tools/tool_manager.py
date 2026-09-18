# ==========================================
# FILE: tools/tool_manager.py
# ==========================================
"""Manage optional workspace custom tools using the fast coder model with validation."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

from ollama import Client

from .config import load_config

TOOLS_DIR = Path("/app/workspace/custom_tools")
TOOLS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = load_config()
FAST_MODEL = CONFIG.get("agent", {}).get("fast_model", "qwen3.5:2b")
FAST_OPTIONS = CONFIG.get("agent", {}).get("fast_options", {"num_ctx": 4096, "temperature": 0.0})
FAST_KEEP_ALIVE = CONFIG.get("worker", {}).get("fast_model_keep_alive", -1)
OLLAMA_HOST = CONFIG.get("agent", {}).get("host", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))


def _fast_client() -> Client:
    return Client(host=os.environ.get("OLLAMA_HOST", OLLAMA_HOST))


def list_tool_files() -> str:
    """List optional custom tool source files in the workspace tool directory."""
    try:
        files = sorted(f for f in os.listdir(TOOLS_DIR) if f.endswith(".py"))
        return f"Custom tool files in {TOOLS_DIR}:\n" + ("\n".join(f"- {f}" for f in files) or "(none)")
    except Exception as exc:
        return f"Error: {exc}"


def _validate_tool_code(code_text: str) -> tuple[bool, str]:
    """Validate python AST syntax and required @agent_tool decorator presence."""
    try:
        tree = ast.parse(code_text)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    has_tool = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "agent_tool":
                    has_tool = True
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "agent_tool":
                    has_tool = True
    if not has_tool:
        return False, "Validation Error: No function decorated with @agent_tool found in the code."

    return True, "Valid"


def create_or_update_tool(tool_name: str, specification: str) -> str:
    """Generate, validate, test-load, and save a custom tool using the fast coder model with self-correction."""
    tool_name = str(tool_name).strip().lower().replace(".py", "").replace(" ", "_")
    if not tool_name:
        return "Error: Missing required 'tool_name' parameter."
    
    filename = f"{tool_name}.py"
    if filename.startswith("_"):
        return "Error: Custom tool filename may not start with '_'."

    file_path = TOOLS_DIR / filename

    system_prompt = (
        "You are an expert Python tool developer for an autonomous AI agent harness. "
        "Write clean, self-contained Python code for a custom agent tool based on the user specification. "
        "Every tool function must be decorated with `@agent_tool` (imported explicitly from `tools.tool_registry`), "
        "include comprehensive type annotations, and have a clear docstring describing its behavior. "
        "CRITICAL IMPORT RULES:\n"
        "- You may only import standard library modules, `requests`, `sqlite3`, `bs4`, `ollama`, and `tools.tool_registry`.\n"
        "- NEVER import non-existent internal helper modules like `tools.utils`.\n\n"
        "Example structural format:\n"
        "```python\n"
        "import sqlite3\n"
        "from tools.tool_registry import agent_tool\n\n"
        "@agent_tool\n"
        "def example_tool(param: str) -> str:\n"
        "    \"\"\"Tool docstring description.\"\"\"\n"
        "    return f'Result: {param}'\n"
        "```\n"
        "Return ONLY valid Python source code inside a ```python markdown block, with no extra conversational prose."
    )

    current_code = ""
    error_feedback = ""
    client = _fast_client()

    # Self-correction loop (up to 3 attempts)
    for attempt in range(1, 4):
        prompt = (
            f"Tool Name: {tool_name}\n"
            f"Specification: {specification}\n"
        )
        if error_feedback:
            prompt += f"\nPrevious attempt failed validation or test loading with this error:\n{error_feedback}\nPlease correct the code and ensure all imports are valid."

        try:
            # Introduce slight temperature variance on retry attempts to escape logic loops
            options = dict(FAST_OPTIONS)
            if attempt > 1:
                options["temperature"] = 0.2

            response = client.generate(
                model=FAST_MODEL,
                prompt=f"{system_prompt}\n\n{prompt}",
                options=options,
                keep_alive=FAST_KEEP_ALIVE,
                think=False,
            )
            raw_text = response.get("response", "")

            # Extract python code block
            if "```python" in raw_text:
                parts = raw_text.split("```python", 1)[1]
                current_code = parts.split("```", 1)[0].strip()
            elif "```" in raw_text:
                parts = raw_text.split("```", 1)[1]
                current_code = parts.split("```", 1)[0].strip()
            else:
                current_code = raw_text.strip()

            # Validate syntax and decorators
            is_valid, msg = _validate_tool_code(current_code)
            if not is_valid:
                error_feedback = msg
                continue

            # Test dynamic import and execution check
            try:
                file_path.write_text(current_code, encoding="utf-8")
                spec = importlib.util.spec_from_file_location(f"custom_test_{tool_name}", file_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Could not load module spec for testing.")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as import_exc:
                file_path.unlink(missing_ok=True)
                raise import_exc

            return f"Successfully generated, validated, and saved custom tool to {file_path}. Call /reload to activate."
        except Exception as exc:
            error_feedback = str(exc)

    return f"Error: Failed to generate a valid custom tool after 3 attempts. Last error: {error_feedback}"


def read_tool_source(filename: str) -> str:
    """Read a custom tool source file."""
    filename = os.path.basename(str(filename))
    path = TOOLS_DIR / (filename if filename.endswith(".py") else filename + ".py")
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc}"


def reload_tools() -> str:
    """Reload the explicit builtin registry and workspace custom tools."""
    from . import AVAILABLE_TOOLS_MAP, load_tools
    count, errors = load_tools()
    text = f"Tools reloaded: {count} active tools.\nActive: {', '.join(AVAILABLE_TOOLS_MAP.keys())}"
    if errors:
        text += "\n\nLoad errors:\n" + "\n".join(f"- {name}: {error}" for name, error in errors.items())
    return text
