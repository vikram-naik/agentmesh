import json

class Validator:
    def __init__(self, model_client=None, rules=None):
        self.model = model_client
        self.rules = rules or []

    def is_done(self, state):
        # Rule-based early validation
        for r in self.rules:
            result = r(state)
            if result.get("done", False):
                return True, result

        # Fallback LLM validation
        if self.model:
            prompt = f"User: {state.user_query}\n\nContext:{json.dumps(state.results)}\n\nDo we have enough info? yes/no"
            out = self.model.generate(prompt, max_tokens=8, temperature=0)
            return out.strip().startswith("y"), {}

        return False, {}
