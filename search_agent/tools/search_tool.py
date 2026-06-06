import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from dataclasses import dataclass
from search_agent.retriever.hybrid import HybridRetriever


@dataclass
class SearchResults:
    title: str
    url: str
    snippet: str
    score: float


class MockSearchTool:
    def search(self, query: str, top_k: int = 3) -> list[SearchResults]:
        q = query.lower()
        if "old man and the sea" in q:
            return [
                SearchResults(
                    title="The Old Man and the Sea",
                    url="mock://old-man-and-the-sea",
                    snippet="The Old Man and the Sea is a novel by Ernest Hemingway.",
                    score=1.0,
                )
            ]
        if "hemingway" in q and "birthplace" in q:
            return [
                SearchResults(
                    title="Ernest Hemingway",
                    url="mock://ernest-hemingway",
                    snippet="Ernest Hemingway was born in Oak Park, Illinois.",
                    score=1.0,
                )
            ]
        return [
            SearchResults(
                title="Mock result",
                url="mock://default",
                snippet=f"No exact mock result found for query: {query}",
                score=0.1,
            )
        ]


class LocalSearchTool:
    def __init__(self, docs_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]
        if docs_path is None:
            docs_path = project_root / "data" / "processed" / "docs.jsonl"

        index_dir = project_root / "data" / "index"
        dense_model_path = project_root.parent / "models" / "Embedding" / "bge-base-zh"

        self.retriever = HybridRetriever(
            docs_path=docs_path,
            dense_model_path=dense_model_path,
            dense_index_path=index_dir / "faiss.index",
            dense_meta_path=index_dir / "dense_docs.json",
        )

    def search(self, query: str, top_k: int = 3) -> list[SearchResults]:
        docs = self.retriever.search(query, top_k=top_k)
        return [
            SearchResults(
                title=doc.title,
                url=f"local://{doc.doc_id}",
                snippet=doc.text,
                score=doc.score,
            )
            for doc in docs
        ]
