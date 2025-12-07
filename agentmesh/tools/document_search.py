# agentmesh/tools/document_search.py
import json
import numpy as np
import faiss
from agentmesh.tools.base import ToolBase
import os

class DocumentSearchTool(ToolBase):
    name = "document_search"

    def __init__(self, index_dir: str):
        self.index_path = os.path.join(index_dir, "index.bin")
        self.docs_path = os.path.join(index_dir, "docs.json")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        self.docs = json.load(open(self.docs_path, "r", encoding="utf-8"))
        self.emb_dim = self.index.d

    def _embed(self, text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.random(self.emb_dim).astype("float32")

    def call(self, query: str, top_k: int = 3):
        vec = self._embed(query).reshape(1, -1)
        D, I = self.index.search(vec, top_k)
        hits = []
        for idx in I[0]:
            if 0 <= idx < len(self.docs):
                hits.append(self.docs[idx])
        return {"ok": True, "hits": hits}
