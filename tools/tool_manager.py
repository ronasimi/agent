import os
import tempfile
import subprocess

TOOLS_DIR = "/app/tools"

def list_tool_files() -> str:
    """List all Python tool files currently in the tools directory."""
    try:
        files = [f for f in os.listdir(TOOLS_DIR) if f.endswith('.py')]
        return f"Tool files in {TOOLS_DIR}:\n" + "\n".join(f"- {f}" for f in files)
    except Exception as e:
        return str(e)

def create_or_update_tool(filename: str, python_code: str) -> str:
    """Create or update a Python tool after verifying syntax with py_compile. 
    Filename must end with .py. Call reload_tools after using this."""
    filename = os.path.basename(filename) # Prevent path traversal
    if not filename.endswith(".py"):
        return "Error: Filename must end with .py"
    
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_file:
        temp_file.write(python_code)
        temp_path = temp_file.name

    try:
        compile_res = subprocess.run(
            ["python", "-m", "py_compile", temp_path],
            capture_output=True,
            text=True
        )

        if compile_res.returncode != 0:
            error_msg = compile_res.stderr or compile_res.stdout
            return (
                f"Syntax validation failed! Tool was NOT saved.\n"
                f"Fix the following syntax error and try again:\n"
                f"{error_msg}"
            )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    filepath = os.path.join(TOOLS_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(python_code)
        return (
            f"Syntax validation passed! Successfully saved tool code to {filepath}.\n"
            f"You MUST now call reload_tools to activate it."
        )
    except Exception as e:
        return str(e)

def read_tool_source(filename: str) -> str:
    """Read the source code of a tool in the tools directory."""
    filename = os.path.basename(filename) # Prevent path traversal
    if not filename.endswith(".py"):
        filename += ".py"
    filepath = os.path.join(TOOLS_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return str(e)

def reload_tools() -> str:
    """Hot-reload all modules in the tools directory, running doctests before activation."""
    from . import load_tools, AVAILABLE_TOOLS_MAP
    
    active_count, errors = load_tools()
    active_tools = list(AVAILABLE_TOOLS_MAP.keys())
    
    response = f"Tools successfully reloaded. Currently active tools ({active_count}): {', '.join(active_tools)}\n"
    
    if errors:
        response += "\nWARNING: The following modules failed testing and were NOT activated:\n"
        for mod_name, err_output in errors.items():
            response += f"\n--- {mod_name}.py Test Failures ---\n{err_output}\n"
            
    return response
