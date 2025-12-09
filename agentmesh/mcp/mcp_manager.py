import asyncio
import logging
import json
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
    """
    Manages connections to MCP servers and provides tool access.
    Async-first implementation.
    """
    def __init__(self, mcp_servers: dict | Iterable[str] | None = None, **client_kwargs):

        if isinstance(mcp_servers, dict):
            servers = mcp_servers
        else:
            mcp_servers = mcp_servers or []
            servers = {f"server_{i}": url for i, url in enumerate(mcp_servers)}

        self.client = MultiServerMCPClient(servers, **client_kwargs)

        self.tools_by_name: Dict[str, Tuple[object, Optional[str]]] = {}
        self.tools_by_server: Dict[str, Dict[str, object]] = {}
        self.tool_definitions_cache: Optional[str] = None

    async def load_tools(self, timeout: Optional[float] = None):
        """
        Asynchronously loads tools from all configured MCP servers.
        """
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
            
            # Case 2 — dict{ server → [tools] }
            elif isinstance(tools_map, dict):
                for server_name, tools in tools_map.items():
                    srv = {}
                    for t in tools:
                        name = getattr(t, "name", None) or getattr(t, "tool_name", None)
                        if not name:
                            name = t.__class__.__name__
                        self.tools_by_name[name] = (t, server_name)
                        srv[name] = t
                    self.tools_by_server[server_name] = srv
            else:
                raise RuntimeError(f"Unexpected get_tools() type: {type(tools_map)}")
            
            # Update definitions cache
            self._update_tool_definitions()
            return True

        if timeout:
            return await asyncio.wait_for(_load(), timeout)
        return await _load()

    def get_tool(self, key: str):
        return self.tools_by_name.get(key, (None, None))

    def list_tools(self):
        return list(self.tools_by_name.keys())

    async def invoke_tool(self, tool_key: str, /, **kwargs):
        """
        Asynchronously invokes a tool.
        """
        tool_obj, server = self.get_tool(tool_key)
        if tool_obj is None:
            raise KeyError(f"Tool not found: {tool_key}")

        # Ensure async invocation
        try:
            if isinstance(tool_obj, StructuredTool):
                return await tool_obj.ainvoke(kwargs)
            elif hasattr(tool_obj, "ainvoke"):
                return await tool_obj.ainvoke(kwargs)
            else:
                # Fallback to sync invoke, wrapped in thread if necessary, 
                # but for pure MCP tools they should support ainvoke
                return await asyncio.to_thread(tool_obj.invoke, kwargs)
        except Exception as e:
            logger.error(f"Error invoking tool {tool_key}: {e}")
            raise e

    def get_tool_definitions(self) -> str:
        """
        Returns a formatted string of tool definitions for LLM prompts.
        """
        if self.tool_definitions_cache is None:
            return "No tools loaded."
        return self.tool_definitions_cache

    def _update_tool_definitions(self):
        """
        Generates schema string for all loaded tools.
        """
        defs = []
        for name, (tool, _) in self.tools_by_name.items():
            # Get description
            desc = getattr(tool, "description", "") or "No description."
            
            # Get args schema if available
            args_schema = "{}"
            if hasattr(tool, "args_schema") and tool.args_schema:
                 try:
                     if isinstance(tool.args_schema, dict):
                         args_schema = json.dumps(tool.args_schema, indent=2)
                     elif hasattr(tool.args_schema, "model_json_schema"):
                         args_schema = json.dumps(tool.args_schema.model_json_schema(), indent=2)
                     elif hasattr(tool.args_schema, "schema"):
                         args_schema = json.dumps(tool.args_schema.schema(), indent=2)
                 except Exception as e:
                     logger.warning(f"Failed to dump schema for {name}: {e}")
            elif hasattr(tool, "get_input_schema"):
                 # LangChain tool might cache it
                 try:
                     schema_obj = tool.get_input_schema()
                     if isinstance(schema_obj, dict):
                        args_schema = json.dumps(schema_obj, indent=2)
                     elif hasattr(schema_obj, "model_json_schema"):
                        args_schema = json.dumps(schema_obj.model_json_schema(), indent=2)
                     elif hasattr(schema_obj, "schema"):
                        args_schema = json.dumps(schema_obj.schema(), indent=2)
                 except Exception as e:
                     logger.warning(f"Failed to get input schema for {name}: {e}")

            defs.append(f"Tool: {name}\nDescription: {desc}\nArguments Schema: {args_schema}\n")
        
        if not defs:
            self.tool_definitions_cache = "No tools available."
        else:
            self.tool_definitions_cache = "\n".join(defs)
