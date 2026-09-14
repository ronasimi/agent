import os
import subprocess

WORKSPACE_DIR = "/app/workspace"

def execute_shell(command: str) -> str:
    """Execute a bash/shell command in the workspace directory."""
    try:
        result = subprocess.run(command, shell=True, cwd=WORKSPACE_DIR, 
                                capture_output=True, text=True, timeout=15, 
                                stdin=subprocess.DEVNULL)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out."
    except Exception as e:
        return str(e)

def execute_python(code: str) -> str:
    """Execute Python code natively within the workspace and return the output."""
    try:
        script_path = os.path.join(WORKSPACE_DIR, "temp_exec.py")
        with open(script_path, 'w') as f:
            f.write(code)
        
        result = subprocess.run(["python", "temp_exec.py"], cwd=WORKSPACE_DIR, 
                                capture_output=True, text=True, timeout=30,
                                stdin=subprocess.DEVNULL)
        
        if os.path.exists(script_path):
            os.remove(script_path)
            
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Python script execution timed out after 30 seconds."
    except Exception as e:
        return str(e)
