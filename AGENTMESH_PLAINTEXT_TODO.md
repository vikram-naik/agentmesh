AGENTMESH TODO LIST (MASTER BACKLOG)
Version 1.0 — canonical project roadmap

A. Observability, Logging & Debugging
- Node-level structured debug logging:
  Log before/after each node: planner, router, executor, validator, composer
  Log inputs and outputs
  Log LLM prompts, responses, token counts
  Log latency per node
  Save trace to memory and optionally a JSON file

- Execution timeline export:
  Export trace to Chrome Tracing format (.json)
  Support tree view of execution steps

- Hardened error handling:
  Graceful handling of planner/router/executor failures
  Automatic JSON repair for planner
  Retry mechanisms with backoff

B. Visualization UI
- Real-time execution viewer
- Static DAG visualization
- Tool registry visualization:
  MCP tool discovery
  Schema viewer
  Health status indicators

C. MCP-First Tool Architecture
- Replace local Python tools with MCPToolProxy
- Dynamic tool discovery from MCP servers
- Asynchronous executor with streaming support
- Support AgentMesh-to-AgentMesh orchestration via MCP

D. Planner Enhancements
- Planner should understand MCP tool schemas:
  Provide JSON schemas to planner
  Prevent hallucinated arguments

- Stable JSON planning:
  Use few-shot examples
  Auto-correct JSON
  Validate JSON against schemas

- Sequential/parallel/dependency-aware TODOs:
  Planner should output dependencies and execution modes

- Mini workflow DSL:
  Example:
    search(query) -> result1
    extract_keywords(result1.text) -> result2
    summarize(result2)

E. Execution Engine
- DAG-aware TODO executor
- Parallel execution support
- Persistent state and checkpointing
- Cache LLM and tool results

F. Framework Abstractions
- Define base interfaces:
  Planner, Router, Composer, Executor, Tool, LLM client

- Plugin architecture:
  Tools installed via Python packages or MCP

- Typed models using Pydantic:
  Tool input/output models
  State models per node

G. Multi-Agent Architecture
- Agent-to-Agent interactions via MCP
- Event-driven architecture:
  Agents publish/subscribe to topics (Kafka/Redis/NATS)

- Global and domain-scoped tool registries

H. Model Runtime Enhancements
- True llama.cpp token streaming
- Multi-backend support:
  OpenAI, Groq, vLLM, Ollama, llama.cpp

- Multi-model orchestration:
  Small model for router
  Medium model for planner
  Large model for composer

I. Examples & Integrations
- LangChain example
- Autogen example
- CrawlAI example
- DSPy example
- Bare-metal Python example
- Jinja2 template-driven workflow example

J. Developer Experience
- CLI Tool: "agentmesh run --query ..."
- Config loader (YAML/JSON/ENV)
- Utility scripts
- CI pipeline:
  lint, test, docs build

K. Production Hardening
- Rate limiting (per tool / per agent / per tenant)
- Secure MCP communication (TLS, JWT)
- Observability exporters (Prometheus, OpenTelemetry)
- Multi-tenancy support:
  Namespaced tool registries
  Context isolation
