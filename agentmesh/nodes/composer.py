# agentmesh/nodes/composer.py
"""
Composer node implementation.
Async implementation.
"""

import json
import logging
from typing import Optional, Dict, Any

from agentmesh.nodes.base import BaseNode
from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.config import AgentMeshConfig
from agentmesh.runtimes.base_client import ModelClient

logger = logging.getLogger("agentmesh.nodes.composer")

class Composer(BaseNode):
    """
    Composer that summarizes results via an LLM.
    """

    def __init__(
        self, 
        llm: ModelClient, 
        mcp_manager: Optional[MCPManager] = None,
        config: Optional[AgentMeshConfig] = None
    ):
        super().__init__(mcp_manager, config)
        self.llm = llm

    async def compose(self, state: Dict[str, Any]) -> str:
        """
        Generates the final answer.
        """
        user_query = state.get("user_query", "")
        results = json.dumps(state.get("results", {}), indent=2)
        
        # Get prompt from config or default
        template = self.config.get("composer.prompt_template")
        if not template:
            template = (
                "You are a summarizer agent.\n\n"
                "User Query:\n{user_query}\n\n"
                "Collected Tool Results:\n{results}\n\n"
                "Produce a concise final answer."
            )

        prompt = template.format(
            user_query=user_query,
            results=results,
        )
        
        try:
            if hasattr(self.llm, "agenerate"):
                raw = await self.llm.agenerate(prompt, max_tokens=300, temperature=0.2)
            else:
                raw = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        except Exception as e:
            logger.error(f"Composer LLM failed: {e}")
            return "Error generating final answer."

        if isinstance(raw, dict):
            text = raw.get("text", "")
        else:
            text = str(raw)

        return text.strip()
