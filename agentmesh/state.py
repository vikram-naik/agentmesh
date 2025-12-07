from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time, uuid

@dataclass
class Todo:
    id: str
    task: str
    meta: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0

@dataclass
class AgentState:
    id: str
    user_query: str
    todos: List[Todo] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    loops: int = 0
    max_loops: int = 3
    final_answer: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def add_todo(self, task: str, meta: Dict[str,Any]=None):
        self.todos.append(Todo(id=str(uuid.uuid4()), task=task, meta=meta or {}))
