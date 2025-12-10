"""
Router Node
-----------
Role: TACTICAL REFINER / TOOL EXPERT

The Router's responsibility is to bridge the gap between High-Level Intent (Planner) and Specific Tool Execution (Executor).
While the Planner says "Book a flight", the Router knows that effectively means:
1. Call `auth_service` to get token.
2. Call `flight_search_api` with token.

Future Vision:
- This node will handle Dependency Resolution and Tool Choreography.
- It can expand a single "Todo" into a sequences of atomic Tool Calls.
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

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the Router logic within the graph.
        Decides on tactical steps (Routing) and checks for tactical completion.
        """
        todos = state.get("todos", [])
        results = state.get("results", {})
        
        routed_items = []
        
        # 1. Route pending items
        for t in todos:
            # Check if this todo is "done" (result exists) or already routed.
            # If the architecture clears 'todos' after execution, we just route what we see.
            route = await self._route_single(t)
            routed_items.append({"todo": t, "route": route})

        # 2. Determine Tactical Status
        if routed_items:
            status = "CONTINUE"
        else:
            # Queue is empty. Check if we have results that might trigger a tactical follow-up.
            new_routes = await self._check_tactical_followup(state)
            if new_routes:
                routed_items.extend(new_routes)
                status = "CONTINUE"
            else:
                status = "DONE"
        
        return {
            "todos": routed_items, 
            "tactical_status": status
        }

    async def _check_tactical_followup(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Internal: Checks if recent results warrant immediate follow-up actions (Tactical Loop).
        Uses a layered parsing approach:
        1. Try direct JSON parse
        2. Retry with format feedback if parsing fails
        3. Raise clear error if still fails (forces prompt improvement)
        """
        if not self.model: 
            return []
            
        results = state.get("results", {})
        if not results:
            return []
            
        # Simplified Prompt for Tactical Expansion
        template = self.config.get("router.tactical_expansion_template")
        if not template:
            template = (
                "You are a Tactical Router.\n"
                "Analyze the recent tool results. Do we need to run more tools to get specific details?\n"
                "If yes, return a list of tool calls.\n"
                "If no (we have all info or results are final), return []\n\n"
                "Available Tools: {tool_definitions}\n\n"
                "Current Results: {results}\n\n"
                "Return ONLY JSON: [{{\"tool\": \"...\", \"args\": {{...}}}}]"
            )
        
        tool_defs = self.mcp_manager.get_tool_definitions() if self.mcp_manager else 'None'
        user_query = state.get("user_query", "Unknown query")
        
        prompt = template.format(
            tool_definitions=tool_defs,
            results=json.dumps(results, indent=2),
            user_query=user_query
        )
        
        # Attempt 1: Direct LLM call
        raw_response = await self._llm_generate(prompt)
        parsed = self._try_parse_json_response(raw_response)
        
        if parsed is not None:
            return self._wrap_as_routed_items(parsed)
        
        # Attempt 2: Retry with format feedback (load from config, reuse tool_defs)
        logger.info("Tactical expansion: First parse failed. Retrying with format feedback.")
        
        retry_template = self.config.get("router.tactical_retry_template")
        if not retry_template:
            retry_template = (
                "Your previous response was not valid JSON. "
                "Please respond with ONLY a JSON array, no explanation or prose.\n\n"
                "REMINDER - Original question: {user_query}\n\n"
                "Available Tools: {tool_definitions}\n\n"
                "Based on current results, decide:\n"
                "- If more tools needed: [{{\"tool\": \"<tool_name>\", \"args\": {{...}}}}]\n"
                "- If done: []\n\n"
                "Return ONLY the JSON array:"
            )
        
        retry_prompt = retry_template.format(
            user_query=user_query,
            tool_definitions=tool_defs
        )
        retry_response = await self._llm_generate(retry_prompt)
        parsed = self._try_parse_json_response(retry_response)
        
        if parsed is not None:
            return self._wrap_as_routed_items(parsed)
        
        # Attempt 3: Fail clearly - developer needs to improve prompt
        error_msg = (
            f"Tactical expansion failed after retry. LLM did not return valid JSON.\n"
            f"First response: {raw_response[:300]}...\n"
            f"Retry response: {retry_response[:300]}...\n"
            f"Action: Improve the 'router.tactical_expansion_template' prompt to enforce JSON output."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    async def _llm_generate(self, prompt: str) -> str:
        """Helper to call LLM and extract text response."""
        if hasattr(self.model, "agenerate"):
            raw = await self.model.agenerate(prompt, max_tokens=256, temperature=0)
        else:
            raw = self.model.generate(prompt, max_tokens=256, temperature=0)
        return raw["text"] if isinstance(raw, dict) else str(raw)
    
    def _try_parse_json_response(self, text: str) -> Optional[Any]:
        """
        Attempts to parse JSON from LLM response.
        Returns parsed data if successful, None if parsing fails.
        """
        if not text or not text.strip():
            return []
        
        clean_text = text.strip()
        
        # Handle empty array explicitly
        if clean_text == "[]":
            return []
        
        # Try direct parse first (fast path for well-behaved LLMs)
        try:
            data = json.loads(clean_text)
            if isinstance(data, (list, dict)):
                return data
        except json.JSONDecodeError:
            pass
        
        # Try stripping markdown code fences
        if "```" in clean_text:
            # Remove ```json and ``` markers
            stripped = clean_text.replace("```json", "").replace("```", "").strip()
            try:
                data = json.loads(stripped)
                if isinstance(data, (list, dict)):
                    return data
            except json.JSONDecodeError:
                pass
        
        # All attempts failed
        return None
    
    def _wrap_as_routed_items(self, data: Any) -> List[Dict[str, Any]]:
        """Wraps parsed JSON data as routed items with unique task names."""
        if not data:
            return []
            
        if isinstance(data, list):
            routed = []
            for idx, item in enumerate(data):
                if isinstance(item, str):
                    tname = f"tactical_{item}_{idx}"
                    routed.append({"todo": {"task": tname}, "route": {"tool": item, "args": {}}})
                elif isinstance(item, dict) and "tool" in item:
                    tname = f"tactical_{item.get('tool', 'tool')}_{idx}"
                    routed.append({"todo": {"task": tname}, "route": item})
            return routed
        elif isinstance(data, dict) and "tool" in data:
            return [{"todo": {"task": f"tactical_{data['tool']}"}, "route": data}]
        
        return []


    async def _route_single(self, todo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal: Refines a single Todo into an executable Route.
        """
        todo_args = todo.get("args", {}) or {}
        task_raw = (todo.get("task") or "").strip()
        task_norm = task_raw.lower()

        # 1. MCP Tool Direct Match (Fastest)
        r = self._via_mcp(task_norm, todo_args)
        if r:
            logger.info(f"Routed '{task_raw}' via MCP match to '{r['tool']}'")
            return r

        # 2. LLM Routing
        if self.model:
            r = await self._via_llm(todo, todo_args)
            if r:
                logger.info(f"Routed '{task_raw}' via LLM to '{r['tool']}'")
                return r

        # 3. Default Fallback
        return {"tool": "unknown", "args": {"original_task": task_raw, "args": todo_args}}

    async def _via_llm(self, todo: Dict[str, Any], todo_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Use config prompt if available
        template = self.config.get("router.prompt_template")
        if not template:
             template = (
                "Route this TODO into a valid tool.\n"
                "Return ONLY JSON: {{ \"tool\": \"<tool_name>\", \"args\": {{...}} }}\n"
                "Available Tools: {tool_definitions}\n\n"
                "Task: {todo}"
             )
        
        tool_defs = self.mcp_manager.get_tool_definitions() if self.mcp_manager else "None"
        
        prompt = template.format(
            todo=json.dumps(todo, indent=2),
            tool_definitions=tool_defs
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
        """
        Internal heuristic: match task name to tool name.
        """
        if not self.mcp_tools:
            if self.mcp_manager:
                self.mcp_tools = self.mcp_manager.list_tools()
            if not self.mcp_tools:
                return None

        # Exact match
        for tname in self.mcp_tools:
            if tname.lower() == task:
                return {"tool": tname, "args": todo_args}

        # Tail match
        candidates = [t for t in self.mcp_tools if t.split(".")[-1].lower() == task]
        if len(candidates) == 1:
            return {"tool": candidates[0], "args": todo_args}
            
        return None
