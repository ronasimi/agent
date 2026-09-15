import os
import subprocess

WORKSPACE_DIR = "/app/workspace"

def execute_shell(command: str = "") -> str:
    """Execute a bash/shell command in the workspace directory.
    
    Args:
        command: The exact bash command string to execute (REQUIRED).
    """
    if not command or not str(command).strip():
        return "Error: Missing required 'command' parameter. You must provide a valid bash command string, e.g., execute_shell(command='date')"
    
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
    """Execute Python code natively within the workspace and return the output.
    
    Args:
        code: The Python source code string to execute (REQUIRED).
    """
    if not code or not str(code).strip():
        return "Error: Missing required 'code' parameter. You must provide valid Python code string."
        
    try:
        script_path = os.path.join(WORKSPACE_DIR, "temp_exec.py")
        with open(script_path, 'w') as f:
            f.write(code)
        
        result = subprocess.run(["python", "temp_exec.py"], cwd=WORKSPACE_DIR, 
                                capture_output=True, text=True, timeout=30,
                                stdin=subprocess.DEVNULL)
        
        if os.path.exists(script_path):
            os.remove(script_path)
            
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return output.strip() or "Python execution completed with no output."
    except subprocess.TimeoutExpired:
        return "Error: Python script execution timed out after 30 seconds."
    except Exception as e:
        return f"Python execution error: {str(e)}"
