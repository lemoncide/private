import os
from typing import Literal

def _get_sandbox_path() -> str:
    # Ensure sandbox directory exists
    sandbox_dir = os.path.join(os.getcwd(), "sandbox")
    if not os.path.exists(sandbox_dir):
        os.makedirs(sandbox_dir)
    return sandbox_dir

def _validate_path(path: str) -> str:
    """Validate and resolve path to ensure it's within sandbox."""
    sandbox_dir = _get_sandbox_path()
    
    # Handle absolute paths: must start with sandbox_dir
    # Handle relative paths: join with sandbox_dir
    if os.path.isabs(path):
        resolved_path = os.path.normpath(path)
    else:
        resolved_path = os.path.normpath(os.path.join(sandbox_dir, path))
    
    if not resolved_path.startswith(sandbox_dir):
        raise ValueError(f"Access denied: Path '{path}' is outside sandbox directory.")
    
    return resolved_path

def read_file(path: str) -> str:
    """Read content from a file within the sandbox."""
    full_path = _validate_path(path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {path}")
        
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str, mode: Literal["write", "append"] = "write") -> str:
    """Write content to a file within the sandbox."""
    full_path = _validate_path(path)
    
    # Ensure parent directory exists
    parent_dir = os.path.dirname(full_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        
    file_mode = "w" if mode == "write" else "a"
    
    with open(full_path, file_mode, encoding="utf-8") as f:
        f.write(content)
        
    return f"Successfully wrote to {path}"
