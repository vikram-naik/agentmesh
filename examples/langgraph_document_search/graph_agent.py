"""
Graph builder for AgentMesh.
Receives planner/router/executor/validator/composer and
returns a compiled LangGraph state machine.
"""

from typing import Any, Dict
from langgraph.graph import StateGraph, START, END


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
    for todo in state.get("todos", []):
        routed.append({"todo": todo, "route": router.route(todo)})
    return {
        "todos": routed,
        "results": state.get("results", {}),
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }


def executor_node(state: Dict[str, Any], executor):
    results = dict(state.get("results", {}))
    for item in state.get("todos", []):
        todo = item["todo"]
        route = item["route"]
        out = executor.execute(route)
        results[todo["task"]] = out
    return {
        "results": results,
        "todos": [],
        "loops": state.get("loops", 0),
        "max_loops": state.get("max_loops", 3),
    }


def validator_node(state, validator):
    done, info = validator.is_done(state)
    # Merge validator info into results so planner gets the feedback next loop
    results = dict(state.get("results", {}))
    if info:
        # place validator info under a named key to avoid colliding with tool results
        results["_validator"] = info
    return {
        "done": bool(done),
        "results": results,
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


def build_agentmesh_graph(planner, router, executor, validator, composer, logger):
    """
    logger: NodeLogger instance injected from run_graph_stream.py
    """
    graph = StateGraph(dict)

    # Node registration wrapped with logger
    graph.add_node("planner", logger.wrap("planner", lambda s: planner_node(s, planner)))
    graph.add_node("router", logger.wrap("router", lambda s: router_node(s, router)))
    graph.add_node("executor", logger.wrap("executor", lambda s: executor_node(s, executor)))
    graph.add_node("validator", logger.wrap("validator", lambda s: validator_node(s, validator)))
    graph.add_node("composer", logger.wrap("composer", lambda s: composer_node(s, composer)))
    graph.add_node("increment_loop", logger.wrap("increment_loop", increment_loop_node))

    # Edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "executor")
    graph.add_edge("executor", "validator")

    graph.add_conditional_edges(
        "validator",
        lambda st: "done" if st.get("done") else "not_done",
        {
            "done": "composer",
            "not_done": "increment_loop",
        },
    )

    graph.add_edge("increment_loop", "planner")
    graph.add_edge("composer", END)

    return graph.compile()
