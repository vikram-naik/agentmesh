"""
Composer node implementation.

Composes the final answer from collected results using an LLM.
"""

import json


class Composer:
    """
    Composer that summarizes results via an LLM.
    """

    def __init__(self, llm, prompt_template=None):
        self.llm = llm
        self.prompt_template = prompt_template or """
You are a summarizer agent.

User Query:
{user_query}

Collected Tool Results:
{results}

Produce a concise final answer.
"""

    def compose(self, state):
        user_query = state.get("user_query", "")
        results = json.dumps(state.get("results", {}), indent=2)

        prompt = self.prompt_template.format(
            user_query=user_query,
            results=results,
        )

        raw = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        if isinstance(raw, dict):
            text = raw.get("text", "")
        else:
            text = str(raw)

        return text.strip()
