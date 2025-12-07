# agentmesh/nodes/router.py
import json

class Router:
    def __init__(self, model_client=None, static_map=None):
        self.model = model_client
        self.static_map = static_map or {}

    def route(self, todo):
        # try fine-tuned model if present
        if self.model:
            prompt = f"Map this todo to a tool. Output JSON.\nTodo:\n{json.dumps(todo)}"
            out = self.model.generate(prompt, max_tokens=128, temperature=0)
            return json.loads(out)
        # fallback rule-based
        task = todo.get("task","").lower()
        for key, tool in self.static_map.items():
            if key in task:
                # default arg: pass task text as query
                return {"tool": tool, "args": {"query": todo.get("task")}}
        # default map to document_search
        return {"tool": "document_search", "args": {"query": todo.get("task")}}
