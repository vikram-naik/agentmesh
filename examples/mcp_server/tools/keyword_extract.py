from agentmesh.tools.base import ToolBase

class KeywordExtractTool(ToolBase):
    name = "keyword_extract"

    def __init__(self, llm):
        self.llm = llm

    def call(self, text=None, query=None):
        """
        Accept BOTH:
            call(text="some text")
        AND:
            call(query="some text")

        Router commonly passes query=...
        """
        source = text or query
        if not source:
            return {"ok": False, "keywords": [], "error": "No text provided"}

        prompt = (
            "Extract up to 5 important keywords from the following text.\n"
            "Return ONLY a comma-separated list.\n\n"
            f"Text:\n{source}\n\nKeywords:"
        )

        out = self.llm.generate(prompt, max_tokens=40, temperature=0)
        keywords = [k.strip() for k in out.split(",") if k.strip()]

        return {"ok": True, "keywords": keywords}
