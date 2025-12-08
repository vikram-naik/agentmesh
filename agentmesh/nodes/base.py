from typing import Optional, Any, Dict
from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.config import AgentMeshConfig

class BaseNode:
    """
    Base class for all AgentMesh nodes.
    Provides access to:
    - Shared Resources (MCP Manager)
    - Configuration (AgentMeshConfig)
    - Async execution primitives
    """
    def __init__(self, mcp_manager: Optional[MCPManager] = None, config: Optional[AgentMeshConfig] = None):
        self.mcp_manager = mcp_manager
        self.config = config or AgentMeshConfig()

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for the node. 
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement async run(state)")

    def get_tool_list(self):
        """Helper to get list of tool names from MCP manager."""
        if self.mcp_manager:
            return self.mcp_manager.list_tools()
        return []
