import os

WORKSPACE_DIR = os.path.abspath("/app/workspace")
# Ensure directory creation on import to prevent IO crash during early execution
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def _get_safe_path(filename: str) -> str:
    """Resolve absolute path and guarantee the string remains inside the workspace."""
    safe_path = os.path.abspath(os.path.join(WORKSPACE_DIR, filename.lstrip('/')))
    
    if os.path.commonpath([WORKSPACE_DIR, safe_path]) != WORKSPACE_DIR:
        raise ValueError(f"Path traversal attempt blocked: {filename}")
    return safe_path

def read_file(filename: str = "") -> str:
    """Read and return the text content of a file stored in the workspace directory."""
    if not filename or not str(filename).strip():
        return "Error: Missing required 'filename' parameter."
        
    try:
        safe_path = _get_safe_path(filename)
        with open(safe_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{filename}': {str(e)}"

def write_file(filename: str = "", content: str = "") -> str:
    """Write text content to a file in the workspace directory. Automatically creates missing subdirectories."""
    if not filename or not str(filename).strip():
        return "Error: Missing required 'filename' parameter."
        
    try:
        safe_path = _get_safe_path(filename)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        with open(safe_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing to file '{filename}': {str(e)}"
