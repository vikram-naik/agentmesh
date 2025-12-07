# local_model_server.py
# Simple HTTP server exposing /generate for the local llama-cpp-python model
# Usage: python local_model_server.py --model-path /path/to/qwen3-4b-instruct.gguf

import argparse
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from llama_cpp import Llama

app = FastAPI()
llama = None

class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0

@app.on_event("startup")
def load_model():
    global llama
    # llama will be created in __main__ to allow passing model path easily.
    if llama is None:
        raise RuntimeError("Model not initialized. Start via __main__")

@app.post("/generate")
async def generate(req: GenRequest):
    global llama
    params = dict(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=None,
        echo=False
    )
    # llama_cpp returns bytes in 'choices' structure; use text from first choice
    out = llama.create(**params)
    # The response from llama_cpp create() is a dict with 'choices' -> [{'text': ...}]
    text = "".join(c.get("text", "") for c in out.get("choices", []))
    return {"text": text}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to GGUF model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--n_ctx", type=int, default=2048)
    args = parser.parse_args()

    # Initialize the Llama model (llama-cpp-python)
    llama = Llama(model_path=args.model_path, n_ctx=args.n_ctx)

    print(f"Loaded model {args.model_path}; serving on http://{args.host}:{args.port}/generate")
    uvicorn.run(app, host=args.host, port=args.port)
