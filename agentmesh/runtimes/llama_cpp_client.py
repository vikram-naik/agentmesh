import requests
from agentmesh.runtimes.base_client import ModelClient

class LlamaLocalClient(ModelClient):
    """
    Client for llama.cpp running in OpenAI-compatible mode.
    Calls /v1/chat/completions.
    """

    def __init__(self, base_url="http://localhost:8081", model="qwen3-4b-instruct-Q8"):
        self.url = f"{base_url}/v1/chat/completions"
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

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]

        return ""
