# agentmesh/nodes/composer.py
import json

class Composer:
    def __init__(self, llm, prompt_template=None):
        self.llm = llm
        # fallback template
        self.prompt_template = prompt_template or """
You are a summarizer agent.

User Query:
{user_query}

Collected Tool Results:
{results}

Produce a concise final answer for the user.
"""

    def compose(self, state: dict):
        user_query = state.get("user_query", "")
        results = json.dumps(state.get("results", {}), indent=2)

        prompt = self.prompt_template.format(
            user_query=user_query,
            results=results,
        )

        out = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        return out.strip()
