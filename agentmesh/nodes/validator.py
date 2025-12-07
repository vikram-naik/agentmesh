"""
Validator node implementation.

Decides whether results are sufficient to conclude.
If a model is present it can use the LLM for a yes/no decision and
provide a short JSON object explaining *why* and optionally suggest todos.
"""

import json


class Validator:
    """
    Validator that can use rules or an LLM to determine completion.

    model_client: object with .generate(prompt, max_tokens, temperature) -> dict or string
    rules: list of sync functions taking state -> dict with {"done": bool, ...}
    """

    def __init__(self, model_client=None, rules=None):
        self.model = model_client
        self.rules = rules or []

    def is_done(self, state):
        """
        Returns: (done: bool, info: dict)

        info should contain at least {"reason": "<text>"} when not done or done,
        and may include "todo_hints": [ { "task": "...", "args": {...} }, ... ]
        """

        # First, run any rule-based validators (synchronous)
        for r in self.rules:
            try:
                result = r(state)
            except Exception:
                result = {}
            if isinstance(result, dict) and result.get("done", False):
                # Normalize returned dict
                info = {k: v for k, v in result.items() if k != "done"}
                return True, info

        # If no model is available, return not done (so planner can iterate)
        if not self.model:
            return False, {"reason": "No model configured for validation."}

        # Prepare prompt asking for structured JSON output
        user_query = state.get("user_query", "")
        context = state.get("results", {})

        prompt = (
            "You are a validator for an agentic document-search system.\n\n"
            "Decide whether the collected results are sufficient to answer the user query.\n"
            "Return ONLY a JSON object with the following schema:\n\n"
            "{\n"
            '  "done": true|false,          // whether we can stop\n'
            '  "reason": "<short explanation>",\n'
            '  "todo_hints": [              // optional: suggestions for planner (can be empty)\n'
            '    { "task": "search", "args": { "query": "<term>" } }\n'
            "  ]\n"
            "}\n\n"
            f"User query:\n{user_query}\n\n"
            f"Collected results:\n{json.dumps(context, indent=2)}\n\n"
            "If you cannot produce the JSON exactly, reply with a short answer 'yes' or 'no'."
        )

        raw = self.model.generate(prompt, max_tokens=250, temperature=0.0)

        # Extract string
        if isinstance(raw, dict):
            text = raw.get("text", "") or ""
        else:
            text = str(raw or "")

        text = text.strip()
        # Try to parse JSON (best effort)
        info = {}
        done = False
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                done = bool(parsed.get("done", False))
                # Keep reason and todo_hints if provided
                info = {}
                if "reason" in parsed:
                    info["reason"] = parsed["reason"]
                if "todo_hints" in parsed:
                    info["todo_hints"] = parsed["todo_hints"]
                # include any other keys too
                for k, v in parsed.items():
                    if k not in ("done", "reason", "todo_hints"):
                        info[k] = v
                return done, info
            except Exception:
                # fallthrough to yes/no detection
                pass

        # Fallback: simple yes/no detection from text
        lowered = text.lower()
        if lowered.startswith("y") or "yes" in lowered.split():
            return True, {"reason": "Model replied yes/no fallback."}
        if lowered.startswith("n") or "no" in lowered.split():
            return False, {"reason": "Model replied yes/no fallback."}

        # Last fallback: not done
        return False, {"reason": "Unable to parse validator response; assuming not done."}
