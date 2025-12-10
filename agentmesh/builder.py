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
    output = await planner.run(state) # returns {"todos":List[...]}
    return {
        "todos": output.get("todos", []),
        "results": state.get("results", {}),
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
        "user_query": state.get("user_query", ""),
    }

async def _router_node(state, router: Router):
    # Delegate to Router.run logic
    output = await router.run(state)
    
    # Ensure keys are preserved
    return {
        "todos": output.get("todos", []),
        "results": state.get("results", {}), # preserved
        "tactical_status": output.get("tactical_status", "DONE"),
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
        "user_query": state.get("user_query", ""),
    }

async def _executor_node(state, executor: Executor):
    # Executor methods are a bit unique because they might need to iterate.
    # But now we defined executor.run(state) as well.
    # However, executor.run is defined to "iterate routed items".
    # Let's try to use executor.run(state) if possible.
    
    output = await executor.run(state) 
    # output = {"results": updated_results}
    
    return {
        "results": output.get("results", {}),
        "todos": [], # Clear routed todos after execution
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
        "user_query": state.get("user_query", ""),
    }

async def _validator_node(state, validator: Validator):
    output = await validator.run(state)
    # output = {"done": bool, "validator_info": dict}
    
    done = output.get("done", False)
    info = output.get("validator_info", {})
    
    results = dict(state.get("results", {}))
    results["_validator"] = info
    
    return {
        "done": done,
        "results": results,
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
        "user_query": state.get("user_query", ""),
    }

async def _increment_loop_node(state):
    return {
        "results": state.get("results", {}),
        "todos": [],
        "loops": state.get("loops", 0) + 1,
        "max_loops": state.get("max_loops", 3),
        "user_query": state.get("user_query", ""),
    }

async def _composer_node(state, composer: Composer):
    output = await composer.run(state)
    # output = {"final_answer": str}
    
    return {
        "final_answer": output.get("final_answer"),
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
        
        # New Tactical Loop Logic:
        # Router -> [Conditional: ExecuteMore | Validator]
        
        graph.add_conditional_edges(
            "router",
            lambda st: "continue" if st.get("tactical_status") == "CONTINUE" else "done",
            {"continue": "executor", "done": "validator"}
        )
        
        # Executor -> Router (The tactical feed back loop)
        graph.add_edge("executor", "router")

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
