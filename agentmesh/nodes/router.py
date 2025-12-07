# agentmesh/nodes/router.py
"""
Router for AgentMesh — routes TODOS → tool invocations.

Supports:
- Static routing
- LLM routing
- MCP namespaced routing
- Schema-based routing
"""

import json
from typing import Optional, Dict, List


class Router:
    def __init__(
        self,
        model_client=None,
        static_map: Optional[Dict[str, str]] = None,
        mcp_tools: Optional[List[str]] = None,
        schema_map: Optional[Dict[str, dict]] = None
    ):
        self.model = model_client
        self.static_map = static_map or {}
        self.mcp_tools = mcp_tools or []
        self.schema_map = schema_map or {}

    # ----------------------------------------------------------------------

    def route(self, todo):
        todo_args = todo.get("args", {}) or {}
        task_raw = (todo.get("task") or "").strip()
        task = task_raw.lower()

        # 1) LLM routing
        if self.model:
            r = self._via_llm(todo, todo_args)
            if r:
                return r

        # 2) Schema routing
        r = self._via_schema(todo, todo_args)
        if r:
            return r

        # 3) MCP tool routing (namespaced-aware)
        r = self._via_mcp(task, todo_args)
        if r:
            return r

        # 4) Static map fallback
        merged = dict(todo_args)
        for key, tool in self.static_map.items():
            if key in task:
                if "query" not in merged:
                    merged["query"] = todo_args.get("query", task_raw)
                return {"tool": tool, "args": merged}

        # 5) Last fallback
        if "query" not in merged:
            merged["query"] = task_raw or ""
        return {"tool": "document_search", "args": merged}

    # ----------------------------------------------------------------------
    def _via_llm(self, todo, todo_args):
        prompt = (
            "Route this TODO into a tool.\n"
            "Return ONLY JSON: {\"tool\":..., \"args\":{...}}\n\n"
            f"{json.dumps(todo, indent=2)}"
        )
        raw = self.model.generate(prompt, max_tokens=128, temperature=0)
        text = raw["text"] if isinstance(raw, dict) else str(raw)
        try:
            route = json.loads(text)
            if "args" not in route:
                route["args"] = todo_args
            return route
        except Exception:
            return None

    # ----------------------------------------------------------------------
    def _via_schema(self, todo, todo_args):
        if not self.schema_map:
            return None
        task = (todo.get("task") or "").lower()
        for tname in self.schema_map:
            if tname.lower() in task:
                return {"tool": tname, "args": todo_args}
        return None

    # ----------------------------------------------------------------------
    def _via_mcp(self, task, todo_args):
        if not self.mcp_tools:
            return None

        # Exact or substring match on namespaced names
        for tname in self.mcp_tools:
            if tname.lower() == task or tname.lower() in task:
                return {"tool": tname, "args": todo_args}

        # Tail-match ("document_search")
        candidates = [t for t in self.mcp_tools if t.split(".")[-1].lower() == task]
        if len(candidates) == 1:
            return {"tool": candidates[0], "args": todo_args}
        elif len(candidates) > 1:
            return {"tool": candidates[0], "args": todo_args}

        # Loose tail-contained match
        candidates = [t for t in self.mcp_tools if task in t.split(".")[-1].lower()]
        if candidates:
            return {"tool": candidates[0], "args": todo_args}

        return None
