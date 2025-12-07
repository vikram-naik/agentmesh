# agentmesh/mcp/tool_loader.py
"""
Utilities for discovering and loading tools from one or more MCP servers.
Produces a dictionary mapping tool_name -> MCPToolProxy instance.
"""

from typing import Dict, List
from langchain_mcp_adapters.client import MultiServerMCPClient
from agentmesh.tools.proxy import MCPToolProxy


async def load_mcp_tools_single(client: MultiServerMCPClient) -> Dict[str, MCPToolProxy]:
    """
    Load tools from a single MCP server.
    """
    tools = await client.list_tools()
    out = {}
    for t in tools:
        name = t.name
        out[name] = MCPToolProxy(client=client, name=name)
    return out


async def load_mcp_tools_multiple(clients: Dict[str, MultiServerMCPClient]) -> Dict[str, MCPToolProxy]:
    """
    Load tools from multiple MCP servers.
    Tool names are namespaced: "<server>.<toolname>"
    """
    out = {}
    for server_name, client in clients.items():
        tools = await client.list_tools()
        for t in tools:
            fq_name = f"{server_name}.{t.name}"
            out[fq_name] = MCPToolProxy(client=client, name=t.name)
    return out
