# agentmesh/nodes/validator.py
"""
Validator node implementation.
Async implementation.
"""

import json
import logging
from typing import Optional, List, Dict, Any, Tuple

from agentmesh.nodes.base import BaseNode
from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.config import AgentMeshConfig
from agentmesh.runtimes.base_client import ModelClient

logger = logging.getLogger("agentmesh.nodes.validator")

class Validator(BaseNode):
    """
    Validator that decides whether the current state is sufficient to answer the query.
    """
    def __init__(
        self, 
        model_client: Optional[ModelClient] = None, 
        rules: Optional[List[callable]] = None,
        mcp_manager: Optional[MCPManager] = None,
        config: Optional[AgentMeshConfig] = None
    ):
        super().__init__(mcp_manager, config)
        self.model = model_client
        self.rules = rules or []

    async def validate(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Determines if the task is done.
        Returns: (done: bool, info: dict)
        """
        # 1. Rule-based validation (Synchronous for now)
        for r in self.rules:
            try:
                result = r(state)
                if isinstance(result, dict) and result.get("done", False):
                    info = {k: v for k, v in result.items() if k != "done"}
                    return True, info
            except Exception as e:
                logger.warning(f"Rule validator failed: {e}")

        # 2. LLM Validation
        if not self.model:
            return False, {"reason": "No model configured for validation."}

        prompt = self._build_prompt(state)
        
        try:
            if hasattr(self.model, "agenerate"):
                raw = await self.model.agenerate(prompt, max_tokens=250, temperature=0.0)
            else:
                raw = self.model.generate(prompt, max_tokens=250, temperature=0.0)
        except Exception as e:
            logger.error(f"Validator LLM call failed: {e}")
            return False, {"reason": "LLM failed"}

        text = (raw.get("text", "") if isinstance(raw, dict) else str(raw)).strip()
        
        return self._parse_response(text)

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        user_query = state.get("user_query", "")
        # Truncate results if too large? 
        # Ideally use Context class but for now direct is fine tailored for validation
        context = state.get("results", {})
        
        # Check config for prompt
        template = self.config.get("validator.prompt_template")
        if template:
            return template.format(user_query=user_query, context=json.dumps(context, indent=2))

        # Default prompt (fallback)
        return (
            "You are a validator for an agentic system.\n"
            "Decide whether the collected results are sufficient to answer the user query.\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "done": true|false,\n'
            '  "reason": "explanation",\n'
            '  "todo_hints": [ { "task": "search", "args": { ... } } ]\n'
            "}\n\n"
            f"User query:\n{user_query}\n\n"
            f"Collected results:\n{json.dumps(context, indent=2)}\n"
        )

    def _parse_response(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        # Cleanup markdown
        clean_text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed = json.loads(clean_text)
            done = bool(parsed.get("done", False))
            info = {k: v for k, v in parsed.items() if k != "done"}
            return done, info
        except json.JSONDecodeError:
            # Fallback
            lower = clean_text.lower()
            if "true" in lower or "yes" in lower:
                 return True, {"reason": "Fallback parsing detected yes/true"}
            return False, {"reason": "Failed to parse validator response"}
