import json
from agentmesh.runtimes.base_client import ModelClient

class Planner:
    def __init__(self, llm: ModelClient):
        self.llm = llm
        self.prompt_template = """
You are a workflow planner for an agentic system.

Your role:
- Analyze the user query and context
- Produce a list of TODOS needed to answer the query
- Each TODO must be a JSON object:
    {{
        "task": "<name>",
        "args": {{}}
    }}

Return ONLY a JSON array, nothing else.

User Query:
{user_query}

Context:
{context}
"""

    def plan(self, state):
        prompt = self.prompt_template.format(
            user_query=state.get("user_query", ""),
            context=json.dumps(state.get("results", {}))
        )

        out = self.llm.generate(prompt, max_tokens=300)

        try:
            todos = json.loads(out)
        except Exception:
            # fallback: wrap output into single todo
            todos = [{"task": "unknown", "args": {"raw": out}}]

        return todos
