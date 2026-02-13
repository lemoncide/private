import os
import sys

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("RealFileSystem")

@mcp.tool()
def read_file(path: str) -> str:
    """
    Read the content of a file from the local filesystem.
    """
    try:
        if not os.path.exists(path):
            return f"Error: File not found at {path}"
        
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """
    Write content to a file. Creates directories if they don't exist.
    """
    try:
        # Ensure directory exists
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
            
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

@mcp.tool()
def list_directory(path: str) -> str:
    """
    List files and directories in a given path.
    """
    try:
        if not os.path.exists(path):
            return f"Error: Directory not found at {path}"
            
        items = os.listdir(path)
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@mcp.tool()
def get_file_info(path: str) -> str:
    """
    Get metadata about a file or directory.
    """
    try:
        if not os.path.exists(path):
            return f"Error: Path not found at {path}"
            
        stats = os.stat(path)
        info = {
            "size": stats.st_size,
            "created": stats.st_ctime,
            "modified": stats.st_mtime,
            "is_dir": os.path.isdir(path),
            "is_file": os.path.isfile(path)
        }
        return str(info)
    except Exception as e:
        return f"Error getting file info: {str(e)}"

if __name__ == "__main__":
    # Run the server using stdio transport by default
    mcp.run()
