# agentmesh/mcp/mcp_client.py
"""
Thin MCP client using httpx.

Provides sync and async methods to:
- discover tools: GET /mcp/tools
- call a tool: POST /mcp/tools/{tool}/call
- stream a tool: POST /mcp/tools/{tool}/stream (yielding chunks)
"""

from typing import Any, Dict, Optional, AsyncGenerator
import httpx
import json
import time


class MCPClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        self._async_client = None

    def discover_tools(self) -> Dict[str, Any]:
        """GET /mcp/tools -> returns JSON describing tools"""
        url = f"{self.base_url}/mcp/tools"
        r = self._client.get(url)
        r.raise_for_status()
        return r.json()

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sync call a tool: POST /mcp/tools/{tool_name}/call"""
        url = f"{self.base_url}/mcp/tools/{tool_name}/call"
        r = self._client.post(url, json={"args": args})
        r.raise_for_status()
        return r.json()

    async def _ensure_async(self):
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)

    async def call_tool_async(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Async call a tool"""
        await self._ensure_async()
        url = f"{self.base_url}/mcp/tools/{tool_name}/call"
        r = await self._async_client.post(url, json={"args": args})
        r.raise_for_status()
        return r.json()

    async def stream_tool(self, tool_name: str, args: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Async generator - calls /mcp/tools/{tool_name}/stream and yields text chunks.
        Server should send a chunked response with newline separated JSON/text chunks.
        """
        await self._ensure_async()
        url = f"{self.base_url}/mcp/tools/{tool_name}/stream"
        async with self._async_client.stream("POST", url, json={"args": args}) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                # decode and yield; server may send arbitrary text (we don't require JSON)
                try:
                    text = chunk.decode("utf-8")
                except Exception:
                    text = str(chunk)
                yield text

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass
        if self._async_client:
            import asyncio
            asyncio.ensure_future(self._async_client.aclose())
