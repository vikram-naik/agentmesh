"""
Correct FastMCP standalone server for AgentMesh examples.
Run:
    python -m examples.mcp_server.fastmcp_server
"""

from fastmcp.server import FastMCP
from typing import Dict, AsyncGenerator
import asyncio


mcp_app = FastMCP(
    name="agentmesh-fastmcp",
    version="1.0.0",
)


# ---------------------------------------------------------
# SIMPLE TOOLS
# ---------------------------------------------------------

@mcp_app.tool()
def document_search(query: str) -> Dict:
    """
    Minimal demo tool for document search.
    """
    hits = [
        f"Result 1 about {query}",
        f"Result 2 that includes {query}",
        f"Another document referencing {query}",
    ]
    return {"ok": True, "hits": hits}


@mcp_app.tool()
def keyword_extract(text: str) -> Dict:
    """
    Minimal keyword extractor.
    """
    words = [w.strip(".,").lower() for w in text.split() if len(w) > 3]
    kws = list(dict.fromkeys(words))[:20]
    return {"ok": True, "keywords": kws}



# ---------------------------------------------------------
# SERVER START
# ---------------------------------------------------------

if __name__ == "__main__":
    # Default port
    mcp_app.run(transport="streamable-http", host="0.0.0.0", port=7001)