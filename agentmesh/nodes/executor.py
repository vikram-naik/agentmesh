# agentmesh/nodes/executor.py
"""
Executor node for AgentMesh graph.
- If mcp_urls/config provided, initializes MCPManager and loads tools.
- execute(route) will invoke the target tool via MCPManager.invoke_tool(...)
"""

from typing import Dict, Any, Optional
import logging

from agentmesh.mcp.mcp_manager import MCPManager

logger = logging.getLogger("agentmesh.executor")


class Executor:
    def __init__(self, tools: Dict[str, object] | None = None, mcp_urls: dict | list | None = None, mcp_client_kwargs: dict | None = None):
        """
        - tools: local in-process tools mapping name -> tool_obj (optional, for demos)
        - mcp_urls: mapping or list with MCP server configurations (preferred), e.g. {'server1': 'http://host:port/mcp'} or ['http://...']
        - mcp_client_kwargs: extra kwargs forwarded to MultiServerMCPClient
        """
        self.logger = None
        self.tools = tools or {}
        self.mcp_manager: Optional[MCPManager] = None

        if mcp_urls:
            self.mcp_manager = MCPManager(mcp_urls, **(mcp_client_kwargs or {}))
            # load tools synchronously (example style)
            self.mcp_manager.load_tools()
            # merge loaded tools into tools dict (flat namespace)
            for name, (tool_obj, server) in self.mcp_manager.tools_by_name.items():
                # if collision, mcp tool will override local tool; change behavior if needed
                self.tools[name] = tool_obj

    def execute(self, route: Dict[str, Any]):
        """
        route: { "tool": "<toolname or server:toolname>", "args": {...} }
        Returns: result from tool invocation.
        """
        tool_name = route.get("tool")
        args = route.get("args", {})

        # debug log
        if self.logger:
            try:
                self.logger.log("executor", "tool_start", {"tool": tool_name, "args": args})
            except Exception:
                logger.exception("logger.log() failed in executor tool_start")

        # choose invocation path:
        # prefer MCP manager invoke if present and tool_name exists there
        try:
            if self.mcp_manager:
                tool_obj, server = self.mcp_manager.get_tool(tool_name)
                if tool_obj:
                    result = self.mcp_manager.invoke_tool(tool_name, **args)
                else:
                    # fallback to local tool if exists
                    tool = self.tools.get(tool_name)
                    if not tool:
                        raise KeyError(f"Tool {tool_name} not found (mcp/local).")
                    # try standard call patterns on local tool
                    result = self._call_local_tool(tool, args)
            else:
                # mcp not configured: use local tools mapping
                tool = self.tools.get(tool_name)
                if not tool:
                    raise KeyError(f"Tool {tool_name} not found (local).")
                result = self._call_local_tool(tool, args)

            if self.logger:
                try:
                    self.logger.log("executor", "tool_end", {"tool": tool_name, "result": result})
                except Exception:
                    logger.exception("logger.log() failed in executor tool_end")

            return result

        except Exception as e:
            logger.exception("Executor failed calling tool %s: %s", tool_name, e)
            raise

    def _call_local_tool(self, tool, args):
        """
        Call a local (in-process) tool with fallbacks similar to MCPManager.invoke_tool.
        """
        # common names
        for name in ("run", "invoke", "call", "__call__"):
            fn = getattr(tool, name, None)
            if callable(fn):
                try:
                    if args and (name != "run" and name != "invoke"):
                        return fn(**args)
                    else:
                        if len(args) == 1:
                            return fn(next(iter(args.values())))
                        return fn(**args)
                except TypeError:
                    continue
        # last resort: if tool itself is callable, try direct call
        if callable(tool):
            return tool(**args)
        raise RuntimeError("Unable to invoke local tool (no compatible call signature).")

