"""
Planner node implementation.

Generates a list of TODOs (JSON) based on the user query and context.
The planner exposes `.plan(state)` and expects `self.llm.generate(prompt, ...)`
to return either a string or a dict with {"text": "...", "usage": {...}}.
"""

import json
from agentmesh.runtimes.base_client import ModelClient


class Planner:
    """
    Planner that uses an LLM to generate a list of todos.
    """

    def __init__(self, llm: ModelClient):
        self.llm = llm
        self.prompt_template = """
You are a workflow planner for an agentic system.

Your role:
- Analyze the user query and context
- Produce a list of TODOS needed to answer the query
- Each TODO must be a JSON object like:

    {{
        "task": "search",
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

        raw = self.llm.generate(prompt, max_tokens=300)
        if isinstance(raw, dict):
            text = raw.get("text", "")
        else:
            text = str(raw)

        try:
            todos = json.loads(text)
        except Exception:
            todos = [{"task": "unknown", "args": {"raw": text}}]

        return todos
