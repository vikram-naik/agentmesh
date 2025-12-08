"""
Builds the LangGraph StateGraph for the document-search example using AgentBuilder.
"""

from typing import Any, Dict
from agentmesh.builder import AgentBuilder
from agentmesh.logging_utils import NodeLogger

def build_agentmesh_graph(planner, router, executor, validator, composer, logger=None):
    """
    Refactored to use standard AgentBuilder.
    Note: The example script passes initialized nodes. 
    AgentBuilder usually initializes nodes itself, but we can reuse the manual nodes 
    if we slightly adapted it or if we just reimplement the wiring here using async wrappers.
    
    For this example, since the nodes are already instantiated with custom config/LLMs,
    we will rely on AgentBuilder's compile logic if possible, or just replicate the async wiring here.
    
    Actually, to keep this example simple and working with the new Async infrastructure,
    we can just update the wrappers to be async.
    """
    from langgraph.graph import StateGraph, START, END
    
    if logger is None:
        logger = NodeLogger(enabled=True)

    # ---------------------------
    # Async Node Wrappers
    # ---------------------------

    async def _planner_node(state, planner):
        # Planner.plan is now async
        return {
            "todos": await planner.plan(state),
            "results": state.get("results", {}),
            "loops": state.get("loops", 0),
            "max_loops": state.get("max_loops", 3),
        }

    async def _router_node(state, router):
        routed = []
        for t in state.get("todos", []):
             # Router.route is now async
             r = await router.route(t)
             routed.append({"todo": t, "route": r})
        return {
            "todos": routed,
            "results": state.get("results", {}),
            "loops": state.get("loops", 0),
            "max_loops": state.get("max_loops", 3),
        }

    async def _executor_node(state, executor):
        results = dict(state.get("results", {}))
        for item in state.get("todos", []):
            task = item["todo"].get("task", "unknown")
            # Executor.execute is now async
            out = await executor.execute(item["route"])
            results[task] = out
        return {
            "results": results,
            "todos": [],
            "loops": state.get("loops", 0),
            "max_loops": state.get("max_loops", 3),
        }

    async def _validator_node(state, validator):
        # Validator.validate (renamed from is_done) is now async
        # But wait, did I rename `is_done` to `validate`?
        # Yes, in Step 149 I renamed it to `validate`. 
        # But I should check if I kept backward compatibility or if I should update here.
        # I replaced the content entirely, so `is_done` is gone. `validate` is the new method.
        done, info = await validator.validate(state)
        return {
            "done": done,
            "results": state.get("results", {}),
            "loops": state.get("loops", 0),
            "max_loops": state.get("max_loops", 3),
        }

    async def _composer_node(state, composer):
        # Composer.compose is now async
        answer = await composer.compose(state)
        return {
            "final_answer": answer,
            "results": state.get("results", {}),
            "todos": [],
            "loops": state.get("loops", 0),
        }

    async def _increment_loop_node(state):
        return {
            "results": state.get("results", {}),
            "todos": [],
            "loops": state.get("loops", 0) + 1,
            "max_loops": state.get("max_loops", 3),
        }

    # ---------------------------
    # Graph assembly
    # ---------------------------
    
    # We can use AgentBuilder if we want, but since we are passed existing instances,
    # let's rebuild the graph manually using the new async wrappers.

    graph = StateGraph(dict)

    graph.add_node("planner", logger.wrap_async("planner", lambda s: _planner_node(s, planner)))
    graph.add_node("router", logger.wrap_async("router", lambda s: _router_node(s, router)))
    graph.add_node("executor", logger.wrap_async("executor", lambda s: _executor_node(s, executor)))
    graph.add_node("validator", logger.wrap_async("validator", lambda s: _validator_node(s, validator)))
    graph.add_node("composer", logger.wrap_async("composer", lambda s: _composer_node(s, composer)))
    graph.add_node("increment_loop", logger.wrap_async("increment_loop", _increment_loop_node))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "executor")
    graph.add_edge("executor", "validator")

    graph.add_conditional_edges(
        "validator",
        lambda st: "done" if st.get("done") else "not_done",
        {"done": "composer", "not_done": "increment_loop"},
    )
    
    # Check max loops break
    graph.add_conditional_edges(
        "increment_loop",
         lambda st: "continue" if st.get("loops", 0) < st.get("max_loops", 3) else "stop",
         {"continue": "planner", "stop": "composer"} 
    )

    graph.add_edge("composer", END)

    compiled = graph.compile()
    compiled.logger = logger
    return compiled
