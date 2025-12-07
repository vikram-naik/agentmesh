# agentmesh/nodes/executor.py
import time
import json

class Executor:
    def __init__(self, registry, retries=2, backoff=0.3):
        self.registry = registry
        self.retries = retries
        self.backoff = backoff
        self.logger = None  # can be injected; NodeLogger will wrap node-level entries too

    def execute(self, route):
        tool_name = route["tool"]
        args = route.get("args", {}) or {}
        tool = self.registry[tool_name]

        # log tool invocation start (if logger injected)
        if self.logger:
            try:
                self.logger.log("executor", "tool_start", {"tool": tool_name, "args": args})
            except Exception:
                pass

        attempts = 0
        start_time = time.time()
        while attempts <= self.retries:
            try:
                result = tool.call(**args)

                duration_ms = round((time.time() - start_time) * 1000, 2)
                # log tool invocation end
                if self.logger:
                    try:
                        self.logger.log("executor", "tool_end", {
                            "tool": tool_name,
                            "args": args,
                            "result": result,
                            "duration_ms": duration_ms
                        })
                    except Exception:
                        pass

                return result
            except Exception as e:
                attempts += 1
                time.sleep(self.backoff * attempts)
                if attempts > self.retries:
                    # log failure
                    if self.logger:
                        try:
                            self.logger.log("executor", "tool_error", {
                                "tool": tool_name,
                                "args": args,
                                "error": str(e),
                                "attempts": attempts
                            })
                        except Exception:
                            pass
                    raise
