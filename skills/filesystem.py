import os
from typing import Literal

def _get_sandbox_path() -> str:
    # Ensure sandbox directory exists
    sandbox_dir = os.path.join(os.getcwd(), "sandbox")
    if not os.path.exists(sandbox_dir):
        os.makedirs(sandbox_dir)
    return sandbox_dir

def _validate_path(path: str) -> str:
    """Force all file operations into sandbox directory."""
    sandbox_dir = _get_sandbox_path()

    # 只取文件名，忽略用户给的目录
    filename = os.path.basename(path)

    # 强制写入 sandbox
    resolved_path = os.path.normpath(os.path.join(sandbox_dir, filename))

    return resolved_path


def read_file(path: str) -> str:
    """Read content from a file within the sandbox."""
    full_path = _validate_path(path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {path}")
        
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str = "github_summary.txt", content: str = "", mode: Literal["write", "append"] = "write") -> str:
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
