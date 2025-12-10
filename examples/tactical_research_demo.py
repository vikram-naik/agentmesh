import asyncio
import logging
import json
from unittest.mock import MagicMock, AsyncMock

# Adjust path to include project root
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentmesh.builder import AgentBuilder
from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.logging_utils import NodeLogger

# --- MOCK DATA & TOOLS ---

DB = {
    "projects": [
        {"id": "p1", "name": "Project Alpha", "status": "active"},
        {"id": "p2", "name": "Project Beta", "status": "inactive"},
        {"id": "p3", "name": "Project Gamma", "status": "active"},
    ],
    "details": {
        "p1": {"tech_lead": "Alice", "budget": 1000},
        "p3": {"tech_lead": "Bob", "budget": 2000},
    }
}

async def mock_search(status: str):
    """Tool: search_projects"""
    print(f"  [Tool] search_projects(status='{status}')")
    return [p for p in DB["projects"] if p["status"] == status]

async def mock_lookup(project_id: str):
    """Tool: get_project_details"""
    print(f"  [Tool] get_project_details(project_id='{project_id}')")
    return DB["details"].get(project_id, {"error": "not found"})

# --- MOCK LLM (Deterministic for Demo) ---

class DemoLLM:
    def __init__(self):
        pass

    async def agenerate(self, prompt: str, **kwargs):
        # 1. PLANNER LOGIC
        if "You are a workflow planner" in prompt:
            # User wants active project leads.
            # Plan: "search" is the first logical step.
            return '[{"task": "search_projects", "args": {"status": "active"}}]'

        # 2. ROUTER LOGIC
        if "You are a Tactical Router" in prompt:
            # Check inputs to decide routing
            if '"task": "search_projects"' in prompt:
                return '{"tool": "search_projects", "args": {"status": "active"}}'
            
            # TACTICAL EXPANSION LOGIC
            if "Analyze the recent tool results" in prompt:
                # If results only contain the list (p1, p3) but NO details yet:
                if "Project Alpha" in prompt and "tech_lead" not in prompt:
                    # We need details!
                    return '''[
                        {"tool": "get_project_details", "args": {"id": "p1"}},
                        {"tool": "get_project_details", "args": {"id": "p3"}}
                    ]'''
                # If we already have details (tech_lead exists):
                if "tech_lead" in prompt:
                    return "[]"
            
            return '{"tool": "unknown", "args": {}}'

        # 3. VALIDATOR
        if "You are a validator" in prompt:
            if "Alice" in prompt and "Bob" in prompt:
                 return '{"done": true, "reason": "Found leads Alice and Bob"}'
            else:
                 return '{"done": false, "reason": "Need more info"}'
        
        # 4. COMPOSER
        return "Active projects are Alpha (Lead: Alice) and Gamma (Lead: Bob)."
    
    def generate(self, prompt, **kwargs):
        # Sync fallback
        return asyncio.run(self.agenerate(prompt, **kwargs))


# --- MAIN ---

async def main():
    print(">>> Starting Tactical Research Demo")
    
    # 1. Setup Mock Manager
    mgr = MagicMock(spec=MCPManager)
    mgr.list_tools.return_value = ["search_projects", "get_project_details"]
    mgr.get_tool_definitions.return_value = "search_projects(status), get_project_details(id)"
    
    # Dynamic dispatch for tools
    async def invoke(name, **args):
        if name == "search_projects": return await mock_search(args.get("status"))
        if name == "get_project_details": return await mock_lookup(args.get("id"))
        return "Unknown tool"
    mgr.invoke_tool = invoke

    # 2. Build Agent
    builder = AgentBuilder(logger=NodeLogger(enabled=True))
    builder.mcp_manager = mgr # Inject handling
    
    # Use our Demo LLM
    llm = DemoLLM()
    builder.build_nodes(planner_llm=llm, composer_llm=llm)
    
    # 3. COMPILE
    graph = builder.compile_graph()
    
    # 4. EXECUTE
    initial_state = {
        "user_query": "Who are the tech leads for active projects?",
        "history": [],
        "loops": 0,
        "max_loops": 5
    }
    
    print("\n>>> Graph Execution Start")
    async for event in graph.astream(initial_state):
        for node, data in event.items():
            print(f"\n--- Node: {node} ---")
            if "todos" in data and data["todos"]:
                print(f"  Todos: {len(data['todos'])}")
                for t in data["todos"]:
                    print(f"  - {t}")
            if "results" in data:
                # print(f"  ResultsKeys: {list(data['results'].keys())}")
                pass
            if "tactical_status" in data:
                print(f"  Status: {data['tactical_status']}")
                
    print("\n>>> Demo Complete")

if __name__ == "__main__":
    asyncio.run(main())
