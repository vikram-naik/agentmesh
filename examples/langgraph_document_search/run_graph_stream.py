# examples/langgraph_document_search/run_graph_stream.py
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse

from agentmesh.logging_utils import NodeLogger
from agentmesh.runtimes.llama_cpp_client import LlamaLocalClient
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# your FastMCP server
MCP_SERVERS = {
    "server1": {
        "url": "http://localhost:7001/mcp",
        "transport": "streamable_http"
    }
}
    
TRACE_DB = "agentmesh_traces.db"

# External minimal HTML UI
UI_PATH = Path(__file__).parent / "trace_ui.html"

logger = NodeLogger(True, sqlite_path=TRACE_DB)

# ----------------------------------------------------------------------
# APP SETUP
# ----------------------------------------------------------------------

from contextlib import asynccontextmanager

# ----------------------------------------------------------------------
# APP SETUP
# ----------------------------------------------------------------------

# Initialize Builder and Graph globally
# LLM for planner & composer
llm_url = "http://localhost:8081"
planner_llm = LlamaLocalClient(llm_url)
composer_llm = LlamaLocalClient(llm_url)

# Builder refactor
from agentmesh.builder import AgentBuilder

# Initialize Builder
builder = AgentBuilder(logger=logger)

# Build MCP Manager (shared)
builder.build_mcp_manager(MCP_SERVERS)

# Build Nodes
builder.build_nodes(planner_llm, composer_llm)

# Compile Graph
graph = builder.compile_graph()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure MCP tools are loaded (async)
    if builder.mcp_manager:
        print("Loading MCP tools...")
        try:
             await builder.mcp_manager.load_tools()
             print(f"Loaded tools: {builder.mcp_manager.list_tools()}")
        except Exception as e:
             print(f"Error loading MCP tools: {e}")
    yield

app = FastAPI(lifespan=lifespan)

# ----------------------------------------------------------------------
@app.post("/query")
async def query(payload: dict):
    q = payload.get("query", "")
    state = {"user_query": q, "results": {}, "todos": [], "loops": 0, "max_loops": 3}

    async def run():
        async for event in graph.astream(state):
            if event:
                logger.capture(event)
            yield json.dumps(logger.last_event(), indent=2) + "\n"

    return StreamingResponse(run(), media_type="application/json")

# ----------------------------------------------------------------------
@app.get("/trace")
async def trace():
    return HTMLResponse(UI_PATH.read_text())

# ----------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
