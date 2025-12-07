import yaml
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from agentmesh.agent import AgentMeshEngine
from agentmesh.nodes.planner import Planner
from agentmesh.nodes.router import Router
from agentmesh.nodes.executor import Executor
from agentmesh.nodes.validator import Validator
from agentmesh.nodes.composer import Composer
from agentmesh.tools.document_search import DocumentSearchTool
from agentmesh.runtimes.llama_cpp_client import LlamaCppClient

# --- Load configs ---
with open("agent_config.yaml") as f:
    cfg = yaml.safe_load(f)

# --- Instantiate LLM clients ---
planner_llm = LlamaCppClient()
composer_llm = LlamaCppClient()

# --- Setup planner ---
planner = Planner(
    model_client=planner_llm,
    prompt_template=cfg["planner"]["prompt_template"]
)

# --- Setup router ---
router = Router(
    model_client=None,  # use static map for now
    static_map=cfg["router"]["static_map"]
)

# --- Setup tools ---
vector_client = None  # replace with FAISS/Qdrant wrapper
tools = {
    "document_search": DocumentSearchTool(vector_client)
}

executor = Executor(tools)

# --- Setup validator (simple rule) ---
def rule_has_hits(state):
    for result in state.results.values():
        if result.get("hits"):
            return {"done": True}
    return {"done": False}

validator = Validator(rules=[rule_has_hits])

# --- Compose ---
composer = Composer(
    model_client=composer_llm,
    prompt_template=cfg["composer"]["prompt_template"]
)

# --- AgentMesh engine ---
engine = AgentMeshEngine(planner, router, executor, validator, composer)

# --- FastAPI server ---
app = FastAPI()

@app.post("/query")
async def query(req: Request):
    body = await req.json()
    query = body["query"]
    state = engine.run(query)

    def stream():
        for line in state.final_answer.split("\n"):
            yield line + "\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
