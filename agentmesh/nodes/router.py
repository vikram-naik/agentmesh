# agentmesh/nodes/router.py
"""
Router for AgentMesh — routes TODOS → tool invocations.
Async implementation.
"""

import json
from typing import Optional, Dict, List, Any
import logging

from agentmesh.nodes.base import BaseNode
from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.config import AgentMeshConfig
from agentmesh.runtimes.base_client import ModelClient

logger = logging.getLogger("agentmesh.nodes.router")

class Router(BaseNode):
    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        mcp_manager: Optional[MCPManager] = None,
        config: Optional[AgentMeshConfig] = None
    ):
        super().__init__(mcp_manager, config)
        self.model = model_client
        # Cache tool list for synchronous routing checks if needed, 
        # but pure MCP should rely on manager.
        # We'll fetch tools once at init or rely on manager lookup.
        self.mcp_tools = self.get_tool_list()

    async def route(self, todo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines the route (tool + args) for a given TODO item.
        Async method.
        """
        todo_args = todo.get("args", {}) or {}
        task_raw = (todo.get("task") or "").strip()
        task = task_raw.lower()

        # 1. MCP Tool Direct Match (Fastest)
        # If the task name matches a tool directly, just use it.
        # This is common if the Planner is smart enough to use tool names.
        r = self._via_mcp(task, todo_args)
        if r:
            logger.info(f"Routed '{task}' via MCP match to '{r['tool']}'")
            return r

        # 2. LLM Routing (Fallback or Smart routing)
        if self.model:
            r = await self._via_llm(todo, todo_args)
            if r:
                logger.info(f"Routed '{task}' via LLM to '{r['tool']}'")
                return r

        # 3. Default Fallback
        # If we can't route it, we might default to a search tool if available, 
        # or return an error tool.
        logger.warning(f"Could not route task: {task}. Defaulting to manual/unknown.")
        return {"tool": "unknown", "args": {"original_task": task, "args": todo_args}}

    async def _via_llm(self, todo: Dict[str, Any], todo_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Check config for routing prompt?
        prompt = (
            "Route this TODO into a valid tool.\n"
            "Return ONLY JSON: {\"tool\": \"<tool_name>\", \"args\":{...}}\n"
            f"Available Tools: {self.mcp_manager.list_tools() if self.mcp_manager else 'None'}\n\n"
            f"Task: {json.dumps(todo, indent=2)}"
        )
        
        try:
            # Async generation if possible
            if hasattr(self.model, "agenerate"):
                raw = await self.model.agenerate(prompt, max_tokens=128, temperature=0)
            else:
                raw = self.model.generate(prompt, max_tokens=128, temperature=0)

            text = raw["text"] if isinstance(raw, dict) else str(raw)
            clean_text = text.strip().replace("```json", "").replace("```", "")
            
            route = json.loads(clean_text)
            if "args" not in route:
                route["args"] = todo_args
            return route
        except Exception as e:
            logger.warning(f"LLM routing failed: {e}")
            return None

    def _via_mcp(self, task: str, todo_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.mcp_tools:
            # Refresh if empty?
            if self.mcp_manager:
                self.mcp_tools = self.mcp_manager.list_tools()
            if not self.mcp_tools:
                return None

        # Similar logic to before: exact match -> tail match -> containment
        
        # Exact match
        for tname in self.mcp_tools:
            if tname.lower() == task:
                return {"tool": tname, "args": todo_args}

        # Tail match (e.g. "search" -> "server.search")
        candidates = [t for t in self.mcp_tools if t.split(".")[-1].lower() == task]
        if len(candidates) == 1:
            return {"tool": candidates[0], "args": todo_args}
            
        return None
