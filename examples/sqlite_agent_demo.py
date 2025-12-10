#!/usr/bin/env python3
"""
SQLite Agent Demo - Real-World Tactical Loop Test
--------------------------------------------------
This script demonstrates the AgentMesh tactical loop using a real SQLite MCP server.

The scenario forces a multi-step dependency chain:
1. Agent must discover available tables
2. Agent must understand table schemas
3. Agent must construct and execute the correct SQL query

Prerequisites:
1. Start the SQLite MCP server:
   PYTHONPATH=. .venv/bin/python examples/mcp_server/sqlite_server.py

2. Start your local LLM (expected at http://localhost:8081)

3. Run this script:
   PYTHONPATH=. .venv/bin/python examples/sqlite_agent_demo.py

Complex Query:
"What is the total revenue from Electronics products sold in the West region?"

Expected Tactical Loop:
- Planner: "Need to query database for revenue"
- Router: list_tables() -> finds products, sales
- Router (expansion): describe_table(products), describe_table(sales)
- Router (expansion): run_query(SELECT ... JOIN ... WHERE ...)
- Validator: Confirms result
- Composer: Summarizes answer
"""

import asyncio
import json
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentmesh.builder import AgentBuilder
from agentmesh.logging_utils import NodeLogger
from agentmesh.runtimes.llama_cpp_client import LlamaLocalClient
from agentmesh.config import AgentMeshConfig
from pathlib import Path

# --- Configuration ---
MCP_SERVERS = {
    "sqlite_demo": {
        "url": "http://localhost:7002/mcp",
        "transport": "streamable_http"
    }
}

LLM_URL = "http://localhost:8081"

# Custom config override for SQL-specific prompts
CONFIG_OVERRIDE_PATH = Path(__file__).parent / "sqlite_config.yaml"

# Complex query that requires understanding database structure
USER_QUERY = "What is the total revenue from Electronics products sold in the West region?"

# Alternative queries for testing:
# USER_QUERY = "How many sales were made in each region?"
# USER_QUERY = "What are the top 3 best-selling products by quantity?"
# USER_QUERY = "List all enterprise tier customers and their regions."

async def main():
    print("=" * 60)
    print("SQLite Agent Demo - Tactical Loop Test")
    print("=" * 60)
    print(f"\nQuery: {USER_QUERY}\n")
    
    # --- Setup ---
    logger = NodeLogger(enabled=True)
    
    # Load custom config with SQL-specific prompts
    AgentMeshConfig.reset()  # Reset singleton to allow fresh load
    import os
    os.environ["AGENTMESH_CONFIG_PATH"] = str(CONFIG_OVERRIDE_PATH)
    config = AgentMeshConfig()
    print(f"Loaded config override: {CONFIG_OVERRIDE_PATH}")
    
    # Initialize LLM
    llm = LlamaLocalClient(LLM_URL)
    
    # Build Agent with custom config
    builder = AgentBuilder(logger=logger, config=config)
    builder.build_mcp_manager(MCP_SERVERS)
    builder.build_nodes(planner_llm=llm, composer_llm=llm)
    
    # Load MCP tools
    print("Loading MCP tools...")
    await builder.mcp_manager.load_tools()
    print(f"Available tools: {builder.mcp_manager.list_tools()}")
    print()
    
    # Compile graph
    graph = builder.compile_graph()
    
    # --- Execute ---
    initial_state = {
        "user_query": USER_QUERY,
        "results": {},
        "todos": [],
        "loops": 0,
        "max_loops": 5
    }
    
    print("Starting graph execution...")
    print("-" * 60)
    
    step_count = 0
    final_answer = None
    
    async for event in graph.astream(initial_state, config={"recursion_limit": 50}):
        step_count += 1
        for node_name, data in event.items():
            print(f"\n[Step {step_count}] Node: {node_name}")
            
            # Show tactical status if present
            if "tactical_status" in data:
                status = data["tactical_status"]
                icon = "🔄" if status == "CONTINUE" else "✅"
                print(f"  {icon} Tactical Status: {status}")
            
            # Show todos
            if "todos" in data and data["todos"]:
                print(f"  📋 Todos: {len(data['todos'])}")
                for t in data["todos"][:3]:  # Limit output
                    if isinstance(t, dict):
                        task = t.get("todo", {}).get("task", str(t))
                        route = t.get("route", {}).get("tool", "")
                        print(f"    - {task} -> {route}")
            
            # Show validation result
            if "done" in data:
                icon = "✅" if data["done"] else "❌"
                print(f"  {icon} Validation: done={data['done']}")
            
            # Capture final answer
            if "final_answer" in data:
                final_answer = data["final_answer"]
    
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(final_answer if final_answer else "No answer generated.")
    print()
    
    # Show execution stats
    print(f"Total steps: {step_count}")
    print("Demo complete.")

if __name__ == "__main__":
    asyncio.run(main())
