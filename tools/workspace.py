# ==========================================
# FILE: tools/workspace.py
# ==========================================
import os

WORKSPACE_DIR = os.path.abspath("/app/workspace")

def _get_safe_path(filename: str) -> str:
    """Resolve absolute path and guarantee the string remains inside the workspace.
    By explicitly avoiding realpath, this allows the OS to follow symlinks 
    that originate from within the workspace directory."""
    safe_path = os.path.abspath(os.path.join(WORKSPACE_DIR, filename.lstrip('/')))
    
    if not safe_path.startswith(WORKSPACE_DIR):
        raise ValueError(f"Path traversal attempt blocked: {filename}")
    return safe_path

def read_file(filename: str = "") -> str:
    """Read and return the text content of a file stored in the workspace directory.
    
    Args:
        filename: Name of the file inside /app/workspace to read. Can include subdirectories (e.g. logs/scan.txt).
    """
    if not filename or not str(filename).strip():
        return "Error: Missing required 'filename' parameter."
        
    try:
        safe_path = _get_safe_path(filename)
        with open(safe_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{filename}': {str(e)}"

def write_file(filename: str = "", content: str = "") -> str:
    """Write text content to a file in the workspace directory. Automatically creates missing subdirectories.
    
    Args:
        filename: Target filename inside /app/workspace. Can include subdirectories (e.g. logs/scan.txt).
        content: The text contents to write into the file.
    """
    if not filename or not str(filename).strip():
        return "Error: Missing required 'filename' parameter."
        
    try:
        safe_path = _get_safe_path(filename)
        
        # Ensure deeply nested directories exist before attempting to write
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        with open(safe_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing to file '{filename}': {str(e)}"
