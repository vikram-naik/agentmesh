"""
Planner Node
------------
Role: STRATEGIST / ARCHITECT

The Planner's responsibility is HIGH-LEVEL problem decomposition.
It analyzes the user's intent and breaks it down into coarse-grained "Todos" or goals.
It has visibility into *available* tools but should focus on "What needs to be done" rather than "How strictly to call every API".

Output:
- A list of `Todo` items (intent + high-level arguments).
"""

import json
import re
import logging
from typing import Optional, List, Dict, Any

from agentmesh.runtimes.base_client import ModelClient
from agentmesh.nodes.base import BaseNode
from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.config import AgentMeshConfig
from agentmesh.nodes.context import PlannerContext

logger = logging.getLogger("agentmesh.nodes.planner")

class Planner(BaseNode):
    """
    Planner node implementation.
    """

    def __init__(self, llm: ModelClient, mcp_manager: Optional[MCPManager] = None, config: Optional[AgentMeshConfig] = None):
        super().__init__(mcp_manager, config)
        self.llm = llm

    async def plan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a high-level plan (list of Todos) based on the state.
        Focuses on strategy, not granular API choreography.
        
        Args:
            state: The current state dictionary.
            
        Returns:
            A list of valid TODO items.
        """
        user_query = state.get("user_query", "")
        
        # 1. Prepare Context
        context = PlannerContext.from_state(state)
        formatted_context = context.format_for_prompt()
        
        # 2. Extract feedback
        results = state.get("results", {})
        feedback_str = self._extract_feedback(results)
        
        # 3. List tools
        tools_str = "No specific tools defined."
        if self.mcp_manager:
            tools_str = self.mcp_manager.get_tool_definitions()

        # 4. Construct Prompt
        # Check config for template, fallback to default if not found (though Config ensures defaults)
        template = self.config.get("planner.prompt_template")
        if not template:
            # Fallback hardcoded just in case config fails catastrophic
            template = "You are a planner. Return JSON array of todos. Query: {user_query}. Tools: {tool_definitions}"
            
        prompt = template.format(
            user_query=user_query,
            context=formatted_context,
            feedback=feedback_str,
            tool_definitions=tools_str
        )

        # 5. Generate
        try:
            # Ideally llm.generate would be async. Wrapping safely just in case.
            if hasattr(self.llm, "agenerate"):
                raw_response = await self.llm.agenerate(prompt, max_tokens=1000)
            else:
                # Assuming standard client might be sync
                # We could run in thread, but for now just call it.
                raw_response = self.llm.generate(prompt, max_tokens=1000)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return [{"task": "error", "args": {"reason": "LLM generation failed"}}]

        if isinstance(raw_response, dict):
             text_response = raw_response.get("text", "")
        else:
             text_response = str(raw_response)

        # 6. Parse and Validate
        todos = self._parse_json_response(text_response)
        
        return todos

    def _extract_feedback(self, results: Dict[str, Any]) -> str:
        feedback_str = "None"
        if "_validator" in results:
            val_info = results["_validator"]
            if isinstance(val_info, dict):
                reason = val_info.get("reason", "Unknown reason.")
                hints = val_info.get("todo_hints", [])
                feedback_str = f"Specific Feedback: {reason}\nSuggested Actions: {json.dumps(hints)}"
            else:
                feedback_str = str(val_info)
        return feedback_str

    def _parse_json_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Robustly parses JSON from LLM output, handling markdown blocks.
        """
        # Strip markdown code blocks
        clean_text = text.strip()
        # Regex to find json block
        match = re.search(r"```json\s*(.*?)```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1)
        else:
            # Try finding array brackets [ ... ]
            match = re.search(r"(\[.*\])", clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1)

        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {text[:100]}...")
            return [{"task": "unknown", "args": {"raw": text}}]

        if not isinstance(data, list):
             # Ensure list
             return [{"task": "unknown", "args": {"raw": "Expected list", "got": str(type(data))}}]

        # Validate items
        valid_todos = []
        for item in data:
            if isinstance(item, dict) and "task" in item:
                valid_todos.append(item)
            else:
                 # Skip or mark invalid items
                 pass
                 
        return valid_todos

