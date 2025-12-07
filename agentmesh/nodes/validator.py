"""
Validator node implementation.

Decides whether results are sufficient to conclude.
If a model is present it can use the LLM for a yes/no decision.
"""

import json


class Validator:
    """
    Validator that can use rules or an LLM to determine completion.
    """

    def __init__(self, model_client=None, rules=None):
        self.model = model_client
        self.rules = rules or []

    def is_done(self, state):
        for r in self.rules:
            result = r(state)
            if result.get("done", False):
                return True, result

        if self.model:
            user_query = state.get("user_query", "")
            context = state.get("results", {})

            prompt = (
                f"User query:\n{user_query}\n\n"
                f"Collected results:\n{json.dumps(context, indent=2)}\n\n"
                "Do we have enough information? Reply: yes or no."
            )

            raw = self.model.generate(prompt, max_tokens=8, temperature=0)
            if isinstance(raw, dict):
                text = raw.get("text", "")
            else:
                text = str(raw)

            return text.strip().lower().startswith("y"), {}

        return False, {}
