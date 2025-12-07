# examples/langgraph_document_search/build_faiss_index.py
import faiss
import numpy as np
import json
import os

docs = [
    "India fiscal policy focuses on long term economic growth.",
    "The central bank controls monetary policy.",
    "Japan fiscal stimulus increased during 2020 pandemic.",
    "Technology companies saw revenue growth in 2024.",
    "Inflation affects disposable income.",
    "TCS reported revenue of 2024 was X (example).",
    "Infosys FY2024 revenue was Y (example)."
]

out_dir = "examples/langgraph_document_search/faiss_index"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "docs.json"), "w") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

emb_dim = 128
# For demo: deterministic pseudo-embeddings using hashed RNG
def embed_text(text):
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.random(emb_dim).astype("float32")

embeddings = np.stack([embed_text(d) for d in docs])

index = faiss.IndexFlatL2(emb_dim)
index.add(embeddings)
faiss.write_index(index, os.path.join(out_dir, "index.bin"))

print("Created FAISS index at", out_dir)
