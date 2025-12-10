"""
Executor Node
-------------
Role: DOER / MECHANIC

The Executor's responsibility is pure, stateless ENACTMENT.
It takes a fully formed, clear instruction (Route) and executes it against the environment (MCP).
It should *not* be making complex decisions or routing logic. It simply runs what it is told.
"""

from typing import Dict, Any, Optional, List
import logging

from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.nodes.base import BaseNode
from agentmesh.config import AgentMeshConfig

logger = logging.getLogger("agentmesh.executor")


class Executor(BaseNode):
    """
    Executor node responsible for running tools defined in the route.
    Async implementation.
    """
    def __init__(self, mcp_manager: Optional[MCPManager] = None, config: Optional[AgentMeshConfig] = None):
        super().__init__(mcp_manager, config)

    async def execute(self, route: Dict[str, Any]) -> Any:
        """
        Enacts a single Route.
        Blindly executes the tool provided.
        
        Args:
            route: { "tool": "<toolname>", "args": {...} }
        """
        tool_name = route.get("tool")
        args = route.get("args", {})
        
        logger.info(f"Executor invoking tool: {tool_name} with args: {args}")

        if not self.mcp_manager:
            raise RuntimeError("MCPManager is not configured in Executor.")

        try:
            # MCPManager.invoke_tool is now async and handles lookup
            result = await self.mcp_manager.invoke_tool(tool_name, **args)
            return result

        except Exception as e:
            logger.error(f"Executor failed calling tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard BaseNode run method.
        Expects state to contain 'todos' with routing info or be called iteratively.
        However, usually Executor is called recursively or iteratively by the graph.
        
        If this is the main entry point (standard graph), it might iterate over todos.
        """
        # Note: The standard graph pattern usually splits logic:
        # Planner -> list of todos
        # Router -> list of (todo, route)
        # Executor -> execution of routes
        
        # This run method assumes it receives a list of routed tasks 
        # OR it receives a single route to execute?
        # Let's assume standard graph behavior: processing 'todos' from state.
        
        # NOTE: The current standard graph logic in builder.py seems to call `executor.execute` iteratively.
        # We will keep `execute` as the public API for granular calls, 
        # and implement `run` to handle batch execution from state if needed.
        
        results = state.get("results", {}).copy()
        todos = state.get("todos", [])
        
        for item in todos:
            # item structure depends on Router output. 
            # Assuming: {"todo": {...}, "route": {"tool": "...", "args": ...}}
            route = item.get("route")
            if route:
                task_name = item.get("todo", {}).get("task", "unknown")
                tool_name = route.get("tool", task_name)
                args = route.get("args", {})
                
                # Generate unique key to prevent overwrites
                # Use tool name + short hash of serialized args
                if args:
                    import hashlib
                    import json
                    args_str = json.dumps(args, sort_keys=True)
                    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
                    result_key = f"{tool_name}_{args_hash}"
                else:
                    result_key = tool_name
                
                output = await self.execute(route)
                results[result_key] = output
                
        return {"results": results}

