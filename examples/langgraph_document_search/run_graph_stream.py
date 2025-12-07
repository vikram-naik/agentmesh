# examples/langgraph_document_search/run_graph_stream.py
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse

from agentmesh.logging_utils import NodeLogger
from agentmesh.nodes.planner import Planner
from agentmesh.nodes.router import Router
from agentmesh.nodes.executor import Executor
from agentmesh.nodes.validator import Validator
from agentmesh.nodes.composer import Composer
from examples.langgraph_document_search.graph_agent import build_agentmesh_graph

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

app = FastAPI()

# LLM for planner & composer
llm_url = "http://localhost:8081"
planner_llm = LlamaLocalClient(llm_url)
composer_llm = LlamaLocalClient(llm_url)

# Nodes
planner = Planner(planner_llm)
validator = Validator(planner_llm)
composer = Composer(composer_llm)

# Executor auto-discovers MCP tools
executor = Executor(mcp_urls=MCP_SERVERS)

# Router gets MCP tool list + schema (if later enabled)
router = Router(
    model_client=None,
    static_map={"search": list(executor.tools.keys())[0]},  # fallback
    mcp_tools=list(executor.tools.keys())
)

graph = build_agentmesh_graph(planner, router, executor, validator, composer, logger)
# logger.attach_graph(graph)

# ----------------------------------------------------------------------
@app.post("/query")
async def query(payload: dict):
    q = payload.get("query", "")
    state = {"user_query": q, "results": {}, "todos": [], "loops": 0, "max_loops": 3}

    def run():
        for event in graph.stream(state):
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
