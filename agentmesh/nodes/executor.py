import time

class Executor:
    def __init__(self, registry, retries=2, backoff=0.3):
        self.registry = registry
        self.retries = retries
        self.backoff = backoff

    def execute(self, route):
        tool_name = route["tool"]
        args = route.get("args", {})
        tool = self.registry[tool_name]

        attempts = 0
        while attempts <= self.retries:
            try:
                return tool.call(**args)
            except Exception as e:
                attempts += 1
                time.sleep(self.backoff * attempts)
                if attempts > self.retries:
                    raise e
