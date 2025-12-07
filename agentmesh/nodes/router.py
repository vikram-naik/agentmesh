"""
Router node implementation.

Maps todos to tool routes. If router has a model (model attribute),
it uses LLM-based routing; otherwise falls back to static_map rules.
"""

import json


class Router:
    """
    Router that routes todos to tools.
    """

    def __init__(self, model_client=None, static_map=None):
        self.model = model_client
        self.static_map = static_map or {}

    def route(self, todo):
        todo_args = todo.get("args", {}) or {}

        if self.model:
            prompt = (
                "Route the following TODO to an appropriate tool.\n"
                "Return ONLY JSON: {\"tool\":\"...\",\"args\":{...}}\n\n"
                f"TODO:\n{json.dumps(todo, indent=2)}"
            )

            raw = self.model.generate(prompt, max_tokens=128, temperature=0)
            if isinstance(raw, dict):
                text = raw.get("text", "")
            else:
                text = str(raw)

            try:
                route = json.loads(text)
                if "args" not in route:
                    route["args"] = todo_args
                return route
            except Exception:
                pass

        task = (todo.get("task") or "").lower()
        for key, tool in self.static_map.items():
            if key in task:
                merged = dict(todo_args)
                if "query" not in merged:
                    merged["query"] = todo_args.get("query", todo.get("task", ""))
                return {"tool": tool, "args": merged}

        merged = dict(todo_args)
        if "query" not in merged:
            merged["query"] = todo_args.get("query", todo.get("task", ""))
        return {"tool": "document_search", "args": merged}
