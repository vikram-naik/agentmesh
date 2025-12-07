import requests
from agentmesh.runtimes.base_client import ModelClient

class LlamaLocalClient(ModelClient):
    """
    Client for llama.cpp running in OpenAI-compatible mode.
    Calls /v1/chat/completions.

    Returns structured output:
    {
        "text": "...",
        "usage": {
            "prompt_tokens": int | None,
            "completion_tokens": int | None
        }
    }
    """

    def __init__(self, base_url="http://localhost:8081", model="qwen3-4b-instruct-Q8"):
        self.url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        r = requests.post(self.url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # --- Extract assistant text ---
        text_out = ""
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                text_out = choice["message"]["content"]

        # --- Extract token usage (llama.cpp provides these in OpenAI format) ---
        usage = data.get("usage", {}) or {}

        return {
            "text": text_out,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")
            }
        }
