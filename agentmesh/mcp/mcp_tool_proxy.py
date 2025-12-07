# agentmesh/mcp/mcp_tool_proxy.py
"""
Per-tool proxy that uses MCPRegistry to invoke a tool and expose call/stream methods.
"""
from __future__ import annotations
from typing import Any, Dict, AsyncIterator, Optional
import asyncio
import json

from .mcp_manager import MCPRegistry
from .http_transport import MCPHTTPTransport


class MCPToolProxy:
    def __init__(self, registry: MCPRegistry, tool_name: str):
        self.registry = registry
        self.tool_name = tool_name

    async def call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Non-streaming call. Returns parsed JSON or {"text": ...}.
        """
        conn = self.registry.pick_transport_for_tool(self.tool_name)
        if not conn:
            raise RuntimeError("No MCP transport available for tool: " + self.tool_name)
        out = await conn.call(self.tool_name, inputs, stream=False)
        return out

    async def stream(self, inputs: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Streaming call. Yields text chunks (strings).
        """
        conn = self.registry.pick_transport_for_tool(self.tool_name)
        if not conn:
            raise RuntimeError("No MCP transport available for tool: " + self.tool_name)
        async for chunk in conn.call(self.tool_name, inputs, stream=True):
            yield chunk

    # sync convenience wrapper (runs the async call)
    def sync_call(self, inputs: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(self.call(inputs))
