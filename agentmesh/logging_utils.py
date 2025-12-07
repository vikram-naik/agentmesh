import time
import json
from datetime import datetime

class NodeLogger:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.events = []

    def log(self, node_name, event_type, data=None):
        if not self.enabled:
            return
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "node": node_name,
            "event": event_type,
            "data": data or {}
        }
        self.events.append(entry)
        print(f"[AgentMesh][{node_name}][{event_type}] {json.dumps(data, indent=2)}")

    def wrap(self, node_name, func):
        def wrapped(state, *args, **kwargs):
            start = time.time()
            self.log(node_name, "start", {"state_in": state})
            try:
                result = func(state, *args, **kwargs)
                elapsed = time.time() - start
                self.log(node_name, "end", {
                    "state_out": result,
                    "duration_ms": round(elapsed * 1000, 2)
                })
                return result
            except Exception as e:
                elapsed = time.time() - start
                self.log(node_name, "error", {
                    "error": str(e),
                    "duration_ms": round(elapsed * 1000, 2)
                })
                raise
        return wrapped
