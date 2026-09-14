import os

WORKSPACE_DIR = "/app/workspace"

def read_file(filename: str) -> str:
    """Read a file's contents from the workspace."""
    try:
        with open(os.path.join(WORKSPACE_DIR, filename), 'r') as f:
            return f.read()
    except Exception as e:
        return str(e)

def write_file(filename: str, content: str) -> str:
    """Write string content to a file in the workspace."""
    try:
        with open(os.path.join(WORKSPACE_DIR, filename), 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return str(e)
