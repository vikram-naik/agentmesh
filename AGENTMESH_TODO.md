# AGENTMESH TODO LIST (MASTER BACKLOG)
Version 1.0 — canonical project roadmap

## A. Observability, Logging & Debugging
A1. Node-level structured debug logging:
- Log before/after each node (planner, router, executor, validator, composer)
- Log inputs/outputs
- Log LLM prompt, response, token counts
- Log latency per node
- Store trace in memory and optional JSON file

A2. Execution timeline export:
- Export trace to Chrome Tracing format (.json)
- Support spanning-tree view

A3. Hardened error handling:
- Graceful handling of planner/router/executor errors
- Automatic JSON repair for planner
- Retry mechanism

## B. Visualization UI
B1. Real-time execution viewer:
- Show node transitions, intermediate tool outputs, metrics

B2. Static DAG view:
- Auto-generate LangGraph DAG diagram

B3. Tool registry visualization:
- MCP tool discovery, schema viewer, health status

## C. MCP-First Tool Architecture
C1. Replace local tools with MCPToolProxy
C2. Dynamic tool discovery from MCP
C3. Async executor with streaming support
C4. Sub-agent (AgentMesh→AgentMesh) via MCP

## D. Planner Enhancements
D1. Planner informed by MCP tool schemas
D2. Stable JSON planning with few-shots
D3. Parallel/sequential dependency-aware TODOs
D4. Mini workflow DSL support

## E. Execution Engine
E1. DAG-aware TODO executor
E2. Persistent state & checkpointing
E3. Caching for LLM + tool results

## F. Framework Abstractions
F1. Define base interfaces for Planner/Router/Composer/Executor/Tool/LLM
F2. Plugin architecture
F3. Typed models via Pydantic

## G. Multi-Agent Architecture
G1. Agent-to-Agent orchestration via MCP
G2. Event-driven workflows
G3. Global + domain-scoped tool registries

## H. Model Runtime Enhancements
H1. True llama.cpp token streaming
H2. Multi-backend support
H3. Multi-model orchestration pipeline

## I. Examples & Integrations
I1. LangChain example
I2. Autogen example
I3. CrawlAI example
I4. DSPy example
I5. Bare-metal Python example
I6. Jinja2 templated workflow example

## J. Developer Experience
J1. CLI tool (agentmesh run ...)
J2. Config loader
J3. Repo scripts
J4. CI pipeline

## K. Production Hardening
K1. Rate limiting
K2. Secure MCP communication (TLS/JWT)
K3. Observability exporters (Prometheus, OpenTelemetry)
K4. Multi-tenancy support
