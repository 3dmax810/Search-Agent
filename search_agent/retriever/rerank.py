import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from search_agent.retriever.bm25 import RetrievedDoc


class Reranker:
    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int = 5,
    ):
        return docs[:top_k]
