import os
import subprocess
import tempfile

WORKSPACE_DIR = "/app/workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def execute_shell(command: str = "") -> str:
    """Execute a bash/shell command in the workspace directory."""
    if not command or not str(command).strip():
        return "Error: Missing required 'command' parameter."
    
    try:
        result = subprocess.run(command, shell=True, cwd=WORKSPACE_DIR, 
                                capture_output=True, text=True, timeout=30, 
                                stdin=subprocess.DEVNULL)
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return output.strip() or "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Execution error: {str(e)}"

def execute_python(code: str = "") -> str:
    """Execute Python code natively within the workspace and return the output."""
    if not code or not str(code).strip():
        return "Error: Missing required 'code' parameter."
        
    try:
        with tempfile.NamedTemporaryFile(dir=WORKSPACE_DIR, suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            script_path = f.name
        
        result = subprocess.run(["python", os.path.basename(script_path)], cwd=WORKSPACE_DIR, 
                                capture_output=True, text=True, timeout=30,
                                stdin=subprocess.DEVNULL)
        
        if os.path.exists(script_path):
            os.remove(script_path)
            
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return output.strip() or "Python execution completed with no output."
    except subprocess.TimeoutExpired:
        if 'script_path' in locals() and os.path.exists(script_path):
            os.remove(script_path)
        return "Error: Python script execution timed out after 30 seconds."
    except Exception as e:
        if 'script_path' in locals() and os.path.exists(script_path):
            os.remove(script_path)
        return f"Python execution error: {str(e)}"
