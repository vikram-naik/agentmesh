import asyncio
import logging
from typing import Dict, Iterable, Optional, Tuple

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception as e:
    raise RuntimeError(
        "Please install langchain-mcp-adapters (pip install langchain-mcp-adapters)"
    ) from e

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger("agentmesh.mcp")


class MCPManager:
    def __init__(self, mcp_servers: dict | Iterable[str] | None = None, **client_kwargs):

        if isinstance(mcp_servers, dict):
            servers = mcp_servers
        else:
            mcp_servers = mcp_servers or []
            servers = {f"server_{i}": url for i, url in enumerate(mcp_servers)}

        self.client = MultiServerMCPClient(servers, **client_kwargs)

        self.tools_by_name: Dict[str, Tuple[object, Optional[str]]] = {}
        self.tools_by_server: Dict[str, Dict[str, object]] = {}

    # ----------------------------------------------------------
    def load_tools(self, timeout: Optional[float] = None):
        async def _load():
            logger.info("Fetching tools from MCP servers…")
            tools_map = await self.client.get_tools()

            self.tools_by_name.clear()
            self.tools_by_server.clear()

            # Case 1 — flat list
            if isinstance(tools_map, list):
                server = "mcp"
                srv = {}
                for t in tools_map:
                    name = getattr(t, "name", None) or getattr(t, "tool_name", None)
                    if not name:
                        name = t.__class__.__name__
                    self.tools_by_name[name] = (t, server)
                    srv[name] = t
                self.tools_by_server[server] = srv
                return True

            # Case 2 — dict{ server → [tools] }
            if isinstance(tools_map, dict):
                for server_name, tools in tools_map.items():
                    srv = {}
                    for t in tools:
                        name = getattr(t, "name", None) or getattr(t, "tool_name", None)
                        if not name:
                            name = t.__class__.__name__
                        self.tools_by_name[name] = (t, server_name)
                        srv[name] = t
                    self.tools_by_server[server_name] = srv
                return True

            raise RuntimeError(f"Unexpected get_tools() type: {type(tools_map)}")

        if timeout:
            return asyncio.run(asyncio.wait_for(_load(), timeout))
        return asyncio.run(_load())

    # ----------------------------------------------------------
    def get_tool(self, key: str):
        return self.tools_by_name.get(key, (None, None))

    def list_tools(self):
        return list(self.tools_by_name.keys())

    # ----------------------------------------------------------
    def invoke_tool(self, tool_key: str, /, **kwargs):
        tool_obj, server = self.get_tool(tool_key)
        if tool_obj is None:
            raise KeyError(f"Tool not found: {tool_key}")

        payload = kwargs

        # StructuredTool requires async
        if isinstance(tool_obj, StructuredTool):
            return asyncio.run(tool_obj.ainvoke(payload))

        if hasattr(tool_obj, "ainvoke"):
            return asyncio.run(tool_obj.ainvoke(payload))

        raise RuntimeError("Tool does not support async invocation.")
