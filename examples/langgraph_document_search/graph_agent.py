"""
Builds the LangGraph StateGraph for the document-search example,
injects a NodeLogger, and wraps model clients so LLM calls are logged.
Nodes themselves remain pure.
"""

from typing import Any, Dict
from langgraph.graph import StateGraph, START, END
from agentmesh.logging_utils import NodeLogger


class LoggingModelWrapper:
    """Wraps any LLM client and logs generate() calls."""

    def __init__(self, model_client, logger: NodeLogger, node_name: str):
        self._model = model_client
        self._logger = logger
        self._node_name = node_name

    def generate(self, prompt: str, max_tokens=256, temperature=0.0):
        raw = self._model.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        if isinstance(raw, dict):
            text = raw.get("text", "")
            usage = raw.get("usage")
        else:
            text = str(raw)
            usage = None

        self._logger.log_llm_call(self._node_name, prompt, text, usage)
        return raw


# ---------------------------
# Node wrappers (pure)
# ---------------------------

def planner_node(state: Dict[str, Any], planner):
    todos = planner.plan(state)
    return {
        "todos": todos,
        "results": state.get("results", {}),
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }


def router_node(state: Dict[str, Any], router):
    routed = []
    for t in state.get("todos", []):
        routed.append({
            "todo": t,
            "route": router.route(t)
        })
    return {
        "todos": routed,
        "results": state.get("results", {}),
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }


def executor_node(state: Dict[str, Any], executor):
    results = dict(state.get("results", {}))
    for item in state.get("todos", []):
        task = item["todo"]["task"]
        out = executor.execute(item["route"])
        results[task] = out
    return {
        "results": results,
        "todos": [],
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }


def validator_node(state, validator):
    done, info = validator.is_done(state)
    return {
        "done": done,
        "results": state.get("results", {}),
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }


def increment_loop_node(state):
    return {
        "results": state.get("results", {}),
        "todos": [],
        "loops": state.get("loops", 0) + 1,
        "max_loops": state.get("max_loops", 3),
    }


def composer_node(state, composer):
    answer = composer.compose(state)
    return {
        "final_answer": answer,
        "results": state.get("results", {}),
        "todos": [],
        "loops": state.get("loops", 0),
    }


# ---------------------------
# Model wrapping
# ---------------------------

def _wrap_models(planner, router, validator, composer, logger):
    if getattr(planner, "llm", None):
        planner.llm = LoggingModelWrapper(planner.llm, logger, "planner")

    if getattr(router, "model", None):
        router.model = LoggingModelWrapper(router.model, logger, "router")

    if getattr(validator, "model", None):
        validator.model = LoggingModelWrapper(validator.model, logger, "validator")

    if getattr(composer, "llm", None):
        composer.llm = LoggingModelWrapper(composer.llm, logger, "composer")


# ---------------------------
# Graph assembly
# ---------------------------

def build_agentmesh_graph(planner, router, executor, validator, composer):
    logger = NodeLogger(
        enabled=True,
        keep_trace=True,
        dump_to_sqlite=False,          # off by default
        sqlite_path="agentmesh_traces.db",
    )

    executor.logger = logger   # tool-level logging

    _wrap_models(planner, router, validator, composer, logger)

    graph = StateGraph(dict)

    graph.add_node("planner", logger.wrap("planner", lambda s: planner_node(s, planner)))
    graph.add_node("router", logger.wrap("router", lambda s: router_node(s, router)))
    graph.add_node("executor", logger.wrap("executor", lambda s: executor_node(s, executor)))
    graph.add_node("validator", logger.wrap("validator", lambda s: validator_node(s, validator)))
    graph.add_node("composer", logger.wrap("composer", lambda s: composer_node(s, composer)))
    graph.add_node("increment_loop", logger.wrap("increment_loop", increment_loop_node))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "executor")
    graph.add_edge("executor", "validator")

    graph.add_conditional_edges(
        "validator",
        lambda st: "done" if st.get("done") else "not_done",
        {"done": "composer", "not_done": "increment_loop"},
    )

    graph.add_edge("increment_loop", "planner")
    graph.add_edge("composer", END)

    compiled = graph.compile()
    compiled.logger = logger
    return compiled
