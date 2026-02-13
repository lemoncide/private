from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("MockGitServer")

@mcp.tool()
def git_clone(repo_url: str, target_dir: str) -> str:
    """
    Simulate cloning a git repository.
    """
    return f"Cloned {repo_url} into {target_dir}"

@mcp.tool()
def read_file(path: str) -> str:
    """
    Simulate reading a file.
    """
    return f"Content of {path}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    return f"Wrote {len(content)} chars to {path}"

@mcp.tool()
def search_repositories(query: str) -> str:
    return f"[mock] search_repositories query={query} -> modelcontextprotocol/servers"

@mcp.tool()
def get_file_contents(repo: str, path: str) -> str:
    if path.lower().endswith("readme.md"):
        return f"# README for {repo}\n\nThis is mock content."
    return f"[mock] {repo}:{path}"

if __name__ == "__main__":
    # Run the server using stdio transport by default
    mcp.run()
