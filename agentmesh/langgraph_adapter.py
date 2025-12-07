# agentmesh/langgraph_adapter.py (pseudo)
from langgraph import Graph, Node

graph = Graph("agentmesh")
planner_node = Node("planner", func=planner.plan)
router_node = Node("router", func=router.route, parallel=True)
executor_node = Node("executor", func=executor.execute)
accumulator_node = Node("accumulate", func=accumulate_results)
validator_node = Node("validator", func=validator.is_done)
composer_node = Node("composer", func=composer.compose)

# connect in DAG; allow loop edges from validator NO -> planner
graph.connect(planner_node, router_node)
graph.connect(router_node, executor_node)
graph.connect(executor_node, accumulator_node)
graph.connect(accumulator_node, validator_node)
graph.connect(validator_node, composer_node, condition="done==true")
graph.connect(validator_node, planner_node, condition="done==false")
