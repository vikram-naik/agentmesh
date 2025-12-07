# AgentMesh

AgentMesh is a lightweight, flexible agent orchestration framework built on top of [LangGraph](https://github.com/langchain-ai/langgraph). It provides a structured way to build agentic workflows using a graph-based architecture with specialized nodes for planning, routing, executing, validating, and composing responses.

## Key Features

- **Graph-Based Architecture**: Leveraging LangGraph for stateful, cyclic, and controllable agent workflows.
- **Specialized Nodes**:
    - **Planner**: Decomposes user queries into actionable TODO items.
    - **Router**: Directs tasks to the appropriate tools or sub-agents.
    - **Executor**: Executes tools and returns results.
    - **Validator**: Checks if the task is complete and validates results.
    - **Composer**: Synthesizes the final answer from the accumulated results.
- **Runtime Abstraction**: Supports different LLM backends (currently includes `LlamaCppClient`).
- **MCP Support**: Integration with the Model Context Protocol (MCP) for tool discovery and invocation.

## Installation

```bash
pip install .
```

## Architecture

AgentMesh uses a standard loop for processing requests:

1.  **Planner**: Receives the initial query and context, generating a list of tasks.
2.  **Router**: Maps each task to a specific tool or handler.
3.  **Executor**: Runs the mapped tools and updates the state with results.
4.  **Validator**: Evaluates if the goal has been met.
    - If **Done**: Passes control to the Composer.
    - If **Not Done**: Increments the loop counter and returns to the Planner (or Router) for the next iteration.
5.  **Composer**: Generates the final response for the user.

## Examples

### LangGraph Document Search

Located in `examples/langgraph_document_search`, this example demonstrates a RAG (Retrieval Augmented Generation) workflow using a local MCP server for tools.

#### Structure

-   `run_graph_stream.py`: **Entry Point**. A FastAPI application that serves the agent and streams responses.
-   `graph_agent.py`: Defines the graph structure using `StateGraph`.
-   `config.yaml`: Configuration for prompts and routing maps.
-   `trace_ui.html`: A simple UI for visualizing the trace.

#### How to Run

This example requires running three separate processes:

1.  **MCP Server**: Hosts the tools (document search, keyword extract).
    ```bash
    python -m examples.mcp_server.fastmcp_server
    ```
    *Runs on port 7001 by default.*

2.  **LLM Server**: Start your LlamaCpp or compatible server.
    *Ensure it is running on the URL configured in `run_graph_stream.py` (default: `http://localhost:8081`).*

3.  **Agent Application**: Run the FastAPI app.
    ```bash
    python examples/langgraph_document_search/run_graph_stream.py
    ```
    *Runs on port 9090 by default.*

#### Usage

Once everything is running, you can query the agent via the streaming endpoint:

```bash
curl -X POST http://localhost:9090/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Search for fiscal policy"}'
```

Or visit `http://localhost:9090/trace` to view the trace UI.

#### Code Walkthrough

The `run_graph_stream.py` script sets up the environment:

```python
# 1. Connect to MCP Server
executor = Executor(mcp_urls={"server1": "http://localhost:7001/mcp"})

# 2. Initialize Nodes
planner = Planner(llm_client)
router = Router(mcp_tools=list(executor.tools.keys()))
# ... other nodes ...

# 3. Build Graph & Serve
graph = build_agentmesh_graph(...)
app = FastAPI()
```

The `build_agentmesh_graph` function connects these nodes into a coherent LangGraph workflow.

## Project Layout

-   `agentmesh/`: Core framework code.
    -   `nodes/`: Implementations of Planner, Router, etc.
    -   `runtimes/`: LLM client wrappers.
    -   `mcp/`: Model Context Protocol integration.
-   `examples/`: Example implementations.
