# agentmesh/mcp/http_transport.py
"""
Async HTTP transport adapter for MCP servers using the Option B endpoints:
  POST /handshake   -> returns {"session_id": "...", ...}
  POST /tools       -> returns {"tools": [...]}
  POST /call        -> accepts call payload, returns either immediate JSON or streaming SSE/chunked

This is intentionally lightweight and dependency-minimal (httpx).
"""
from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, Optional
import httpx
import time


class MCPHTTPTransport:
    def __init__(self, base_url: str, timeout: int = 30):
        # base_url should be like "http://localhost:8001" (no trailing path)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self.last_handshake_ts = 0.0

    async def handshake(self) -> Dict[str, Any]:
        """
        POST {base_url}/handshake
        Expect JSON with session_id.
        """
        url = f"{self.base_url}/handshake"
        resp = await self.client.post(url, timeout=self.timeout)
        resp.raise_for_status()
        out = resp.json()
        sid = out.get("session_id") or out.get("session")
        if not sid:
            raise RuntimeError(f"handshake did not provide session_id: {out}")
        self.session_id = sid
        self.last_handshake_ts = time.time()
        return out

    async def list_tools(self) -> Dict[str, Any]:
        """
        POST {base_url}/tools with session info (if required).
        """
        if not self.session_id:
            await self.handshake()
        url = f"{self.base_url}/tools"
        payload = {"session_id": self.session_id}
        resp = await self.client.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    async def call(self, tool_name: str, inputs: Dict[str, Any], stream: bool = False):
        """
        POST {base_url}/call
        If stream=True, this returns an async iterator yielding text chunks.
        Otherwise returns JSON final response.
        Payload includes session_id, tool, inputs.
        """
        if not self.session_id:
            await self.handshake()
        url = f"{self.base_url}/call"
        payload = {"session_id": self.session_id, "tool": tool_name, "inputs": inputs}

        # If we expect streaming (SSE or chunked), request streaming
        headers = {}
        # Prefer server-sent events if available: accept text/event-stream
        if stream:
            headers["Accept"] = "text/event-stream"

        # Perform request
        resp = await self.client.post(url, json=payload, headers=headers, timeout=None)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if stream and "text/event-stream" in content_type:
            # SSE style streaming: yield event.data pieces
            async for line in resp.aiter_lines():
                if line is None:
                    continue
                line = line.strip()
                if not line:
                    continue
                # SSE lines may come like: "data: {...}"
                if line.startswith("data:"):
                    data = line[5:].strip()
                    yield data
                else:
                    # fallback: yield raw line
                    yield line
            return
        elif stream and ("application/octet-stream" in content_type or content_type.startswith("text/")):
            # chunked text stream
            async for chunk in resp.aiter_text():
                if chunk:
                    yield chunk
            return
        else:
            # Non-stream: return parsed JSON (or raw text)
            try:
                return resp.json()
            except Exception:
                return {"text": resp.text}

    async def close(self):
        await self.client.aclose()
