# examples/langgraph_document_search/run_graph_stream.py

import yaml
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

# ✔ Correct import for module execution
from examples.langgraph_document_search.graph_agent import build_agentmesh_graph

from agentmesh.nodes.planner import Planner
from agentmesh.nodes.router import Router
from agentmesh.nodes.executor import Executor
from agentmesh.nodes.validator import Validator
from agentmesh.nodes.composer import Composer

from agentmesh.runtimes.llama_cpp_client import LlamaLocalClient
from agentmesh.tools.document_search import DocumentSearchTool
from agentmesh.tools.keyword_extract import KeywordExtractTool

# Load config
cfg = yaml.safe_load(open("examples/langgraph_document_search/config.yaml"))

# ---- LLM Client Setup ----
llm_base = "http://localhost:8081"

planner_llm = LlamaLocalClient(base_url=llm_base, model="qwen3-4b-instruct-Q8")
composer_llm = LlamaLocalClient(base_url=llm_base, model="qwen3-4b-instruct-Q8")
keyword_llm = LlamaLocalClient(base_url=llm_base, model="qwen3-4b-instruct-Q8")



planner = Planner(planner_llm)
composer = Composer(composer_llm, cfg["composer"]["prompt_template"])
router = Router(static_map=cfg["router"]["static_map"])

# Tools
index_dir = "examples/langgraph_document_search/faiss_index"
doc_tool = DocumentSearchTool(index_dir=index_dir)
kw_tool = KeywordExtractTool(llm=keyword_llm)

tools = {
    "document_search": doc_tool,
    "keyword_extract": kw_tool
}

executor = Executor(tools)

validator = Validator(rules=[
    lambda st: {"done": any(
        isinstance(v, dict) and v.get("hits")
        for v in st.get("results", {}).values()
    )}
])

graph = build_agentmesh_graph(planner, router, executor, validator, composer)

app = FastAPI()

@app.post("/query")
async def query(req: Request):
    body = await req.json()
    user_query = body.get("query", "")

    events = graph.stream({
        "user_query": user_query,
        "results": {},
        "todos": [],
        "loops": 0,
        "max_loops": 3
    })

    def streamer():
        final = ""
        for ev in events:
            if "composer" in ev:
                ans = ev["composer"]["final_answer"]
                for w in ans.split():
                    yield w + " "
                final = ans
        yield "\n"

    return StreamingResponse(streamer(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
