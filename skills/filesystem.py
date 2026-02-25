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
    """
    从 sandbox 中读取文件内容。

    何时用：需要查看 sandbox 里的文本文件内容（例如读取上一步写入的结果文件）。
    输入：path（文件名或相对路径；会被强制映射到 sandbox 内的文件名）。
    输出：文件的完整文本内容（str）。
    典型任务：读取中间结果、读取待处理文本、读取生成的报告。
    """
    full_path = _validate_path(path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {path}")
        
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str = "github_summary.txt", content: str = "", mode: Literal["write", "append"] = "write") -> str:
    """
    向 sandbox 写入文本内容（覆盖或追加）。

    何时用：需要把总结/报告/中间结果落盘到 sandbox，供后续步骤读取或用户查看。
    输入：path（文件名或相对路径；会被强制映射到 sandbox 内的文件名）、content（写入文本）、mode（write 覆盖 / append 追加）。
    输出：写入成功的提示信息（str）。
    典型任务：生成汇总文件、导出分析结果、追加日志式输出。
    """
    full_path = _validate_path(path)
    
    # Ensure parent directory exists
    parent_dir = os.path.dirname(full_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        
    file_mode = "w" if mode == "write" else "a"
    
    with open(full_path, file_mode, encoding="utf-8") as f:
        f.write(content)
        
    return f"Successfully wrote to {path}"
