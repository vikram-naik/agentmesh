# examples/mcp_tools_server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import time
import json

app = FastAPI()

# Minimal tool implementations (reuse your local logic if desired)
@app.get("/mcp/tools")
def list_tools():
    return {
        "tools": [
            {
                "name": "document_search",
                "description": "Search local faiss index",
                "schema": {"query": {"type": "string"}},
                "type": "search"
            },
            {
                "name": "keyword_extract",
                "description": "Extract keywords from text",
                "schema": {"text": {"type": "string"}},
                "type": "nlp"
            }
        ]
    }

@app.post("/mcp/tools/document_search/call")
async def doc_search_call(req: Request):
    body = await req.json()
    args = body.get("args", {})
    q = args.get("query", "")
    # pretend we're searching:
    hits = [
        f"Result 1 about {q}",
        f"Result 2 mentioning {q}",
        f"Another doc referencing {q}"
    ]
    return {"ok": True, "hits": hits}

@app.post("/mcp/tools/keyword_extract/call")
async def keyword_extract_call(req: Request):
    body = await req.json()
    args = body.get("args", {})
    text = args.get("text", "")
    # super simple: split and return unique words (demo only)
    tokens = text.split()
    keywords = list(dict.fromkeys(tokens))[:10]
    return {"ok": True, "keywords": keywords}

@app.post("/mcp/tools/document_search/stream")
async def doc_search_stream(req: Request):
    body = await req.json()
    args = body.get("args", {})
    q = args.get("query", "")
    # stream three chunks
    def generator():
        for i in range(3):
            time.sleep(0.2)
            chunk = json.dumps({"chunk": f"chunk {i} for {q}"}) + "\n"
            yield chunk.encode("utf-8")
    return StreamingResponse(generator(), media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7070)
