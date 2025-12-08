from typing import Any, Dict, Optional, List
import asyncio

from agentmesh.logging_utils import NodeLogger
from agentmesh.mcp.mcp_manager import MCPManager
from agentmesh.config import AgentMeshConfig
from agentmesh.nodes.planner import Planner
from agentmesh.nodes.router import Router
from agentmesh.nodes.executor import Executor
from agentmesh.nodes.validator import Validator
from agentmesh.nodes.composer import Composer

from langgraph.graph import StateGraph, START, END

# --- Standard Async Node Wrappers ---

async def _planner_node(state, planner: Planner):
    return {
        "todos": await planner.plan(state),
        "results": state.get("results", {}),
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }

async def _router_node(state, router: Router):
    routed = []
    # Route each todo
    for t in state.get("todos", []):
         route = await router.route(t)
         routed.append({"todo": t, "route": route})
         
    return {
        "todos": routed,
        "results": state.get("results", {}),
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }

async def _executor_node(state, executor: Executor):
    results = dict(state.get("results", {}))
    # 'todos' here are the routed items from Router
    for item in state.get("todos", []):
        task_name = item["todo"].get("task", "unknown")
        route = item.get("route")
        if route:
            # Execute async
            out = await executor.execute(route)
            results[task_name] = out
            
    return {
        "results": results,
        "todos": [],
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }

async def _validator_node(state, validator: Validator):
    done, info = await validator.validate(state)
    
    # Merge info (reason, hints) into a special key or directly into results?
    # Ideally we put it in results["_validator"] for the Planner to see next loop.
    results = dict(state.get("results", {}))
    results["_validator"] = info
    
    return {
        "done": done,
        "results": results,
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }

async def _increment_loop_node(state):
    return {
        "results": state.get("results", {}),
        "todos": [],
        "loops": state.get("loops", 0) + 1,
        "max_loops": state.get("max_loops", 3),
    }

async def _composer_node(state, composer: Composer):
    final_ans = await composer.compose(state)
    return {
        "final_answer": final_ans,
        "results": state.get("results", {}),
        "todos": [],
        "loops": state.get("loops", 0),
    }


class LoggingModelWrapper:
    """Wraps any LLM client and logs generate() calls."""
    def __init__(self, model_client, logger: NodeLogger, node_name: str):
        self._model = model_client
        self._logger = logger
        self._node_name = node_name

    def generate(self, prompt: str, max_tokens=256, temperature=0.0):
        # This wrapper might need to support async too if we use agenerate
        raw = self._model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        if isinstance(raw, dict):
            text = raw.get("text", "")
            usage = raw.get("usage")
        else:
            text = str(raw)
            usage = None
        self._logger.log_llm_call(self._node_name, prompt, text, usage)
        return raw
        
    async def agenerate(self, prompt: str, max_tokens=256, temperature=0.0):
        if hasattr(self._model, "agenerate"):
            raw = await self._model.agenerate(prompt, max_tokens=max_tokens, temperature=temperature)
        else:
            # Fallback to sync
            raw = self._model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            
        if isinstance(raw, dict):
            text = raw.get("text", "")
            usage = raw.get("usage")
        else:
            text = str(raw)
            usage = None
        self._logger.log_llm_call(self._node_name, prompt, text, usage)
        return raw


class AgentBuilder:
    def __init__(self, logger: Optional[NodeLogger] = None, config: Optional[AgentMeshConfig] = None):
        self.logger = logger or NodeLogger(enabled=False)
        self.config = config or AgentMeshConfig()
        self.mcp_manager: Optional[MCPManager] = None
        
        self.planner: Optional[Planner] = None
        self.router: Optional[Router] = None
        self.executor: Optional[Executor] = None
        self.validator: Optional[Validator] = None
        self.composer: Optional[Composer] = None

    def build_mcp_manager(self, mcp_servers: Dict[str, Any]):
        """Initializes the shared MCP Manager."""
        self.mcp_manager = MCPManager(mcp_servers)
        # Note: load_tools is async. We can't await here in sync init/builder pattern easily
        # unless we make build async. 
        # For now, we expect the user to await mcp_manager.load_tools() externally 
        # OR we provide an async init method.
        # Let's keep it lazy or assume the runtime handles initialization loop.
        return self

    def build_nodes(self, planner_llm, composer_llm, validator_llm=None):
        """Builds all standard nodes sharing the mcp_manager."""
        
        # Planner
        self.planner = Planner(planner_llm, mcp_manager=self.mcp_manager, config=self.config)
        
        # Executor
        self.executor = Executor(mcp_manager=self.mcp_manager, config=self.config)
        
        # Validator (allow separate LLM or reuse planner LLM)
        v_llm = validator_llm or planner_llm
        self.validator = Validator(v_llm, mcp_manager=self.mcp_manager, config=self.config)
        
        # Composer
        self.composer = Composer(composer_llm, mcp_manager=self.mcp_manager, config=self.config)
        
        # Router
        self.router = Router(
            model_client=planner_llm,
            mcp_manager=self.mcp_manager,
            config=self.config
        )
        return self

    def _wrap_models(self):
        """Wraps models with logging wrappers."""
        if self.planner and hasattr(self.planner, "llm"):
             self.planner.llm = LoggingModelWrapper(self.planner.llm, self.logger, "planner")
        if self.router and hasattr(self.router, "model") and self.router.model:
             self.router.model = LoggingModelWrapper(self.router.model, self.logger, "router")
        if self.validator and hasattr(self.validator, "model"):
             self.validator.model = LoggingModelWrapper(self.validator.model, self.logger, "validator")
        if self.composer and hasattr(self.composer, "llm"):
             self.composer.llm = LoggingModelWrapper(self.composer.llm, self.logger, "composer")

    def compile_graph(self):
        """Constructs and compiles the StateGraph."""
        if not all([self.planner, self.router, self.executor, self.validator, self.composer]):
            raise ValueError("Nodes not built. Call build_nodes() first.")

        # logging wrappers
        self._wrap_models()

        graph = StateGraph(dict)

        # Add Nodes (wrapped with logger logic if needed, but lambdas handle invocation)
        graph.add_node("planner", self.logger.wrap_async("planner", lambda s: _planner_node(s, self.planner)))
        graph.add_node("router", self.logger.wrap_async("router", lambda s: _router_node(s, self.router)))
        graph.add_node("executor", self.logger.wrap_async("executor", lambda s: _executor_node(s, self.executor)))
        graph.add_node("validator", self.logger.wrap_async("validator", lambda s: _validator_node(s, self.validator)))
        graph.add_node("composer", self.logger.wrap_async("composer", lambda s: _composer_node(s, self.composer)))
        graph.add_node("increment_loop", self.logger.wrap_async("increment_loop", _increment_loop_node))

        # Edges
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "router")
        graph.add_edge("router", "executor")
        graph.add_edge("executor", "validator")

        graph.add_conditional_edges(
            "validator",
            lambda st: "done" if st.get("done") else "not_done",
            {"done": "composer", "not_done": "increment_loop"},
        )

        # Check loop breaks
        graph.add_conditional_edges(
            "increment_loop",
             lambda st: "continue" if st.get("loops", 0) < st.get("max_loops", 3) else "stop",
             {"continue": "planner", "stop": "composer"} 
        )
        
        graph.add_edge("composer", END)

        compiled = graph.compile()
        # Attach logger to compiled graph for external access if needed
        compiled.logger = self.logger
        return compiled
