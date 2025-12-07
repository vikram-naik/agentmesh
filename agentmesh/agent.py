import time
from agentmesh.state import AgentState

class AgentMeshEngine:
    def __init__(self, planner, router, executor, validator, composer):
        self.planner = planner
        self.router = router
        self.executor = executor
        self.validator = validator
        self.composer = composer

    def run(self, user_query):
        state = AgentState(id=f"agent-{int(time.time())}", user_query=user_query)

        while state.loops < state.max_loops:
            # 1. PLAN
            if not state.todos:
                todos_dict = self.planner.plan(state)
                for t in todos_dict:
                    state.add_todo(t["task"], t.get("meta", {}))

            # 2. EXECUTE TODOS
            for todo in list(state.todos):
                route = self.router.route({"task": todo.task, "meta": todo.meta})
                result = self.executor.execute(route)
                state.results[todo.id] = result
                state.todos.remove(todo)

            # 3. VALIDATE
            done, info = self.validator.is_done(state)
            if done:
                state.final_answer = self.composer.compose(state)
                return state

            # Loop again
            state.loops += 1

        # Fallback: compose partial answer
        state.final_answer = self.composer.compose(state)
        return state
