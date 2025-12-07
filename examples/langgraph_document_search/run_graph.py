# examples/langgraph_document_search/run_graph.py

import yaml
from langgraph.graph import Messages

from agentmesh.nodes.planner import Planner
from agentmesh.nodes.router import Router
from agentmesh.nodes.executor import Executor
from agentmesh.nodes.validator import Validator
from agentmesh.nodes.composer import Composer

from agentmesh.runtimes.llama_cpp_client import LlamaCppClient
from agentmesh.tools.document_search import DocumentSearchTool

from graph_agent import build_agentmesh_graph


# 1. Load config
cfg = yaml.safe_load(open("config.yaml"))

planner_llm = LlamaCppClient()
composer_llm = LlamaCppClient()

planner = Planner(planner_llm, cfg["planner"]["prompt_template"])
router = Router(static_map=cfg["router"]["static_map"])
validator = Validator(rules=[
    lambda state: {"done": any(
        isinstance(v, dict) and v.get("hits")
        for v in state["results"].values()
    )}
])
composer = Composer(composer_llm, cfg["composer"]["prompt_template"])

# Tools
vector_client = None  # Replace with FAISS/Qdrant instance
tool_registry = {
    "document_search": DocumentSearchTool(vector_client)
}
executor = Executor(tool_registry)

# 2. Build graph
graph = build_agentmesh_graph(planner, router, executor, validator, composer)

# 3. Run the agent
query = "Search the document store for fiscal policy actions"
state = graph.invoke({
    "user_query": query,
    "results": {},
    "todos": [],
    "loops": 0,
    "max_loops": 3
})

print("\n=== FINAL ANSWER ===\n")
print(state["final_answer"])
