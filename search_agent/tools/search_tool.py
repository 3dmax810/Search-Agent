import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from dataclasses import dataclass


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
    def __init__(
        self,
        docs_path: str | Path | None = None,
        index_dir: str | Path | None = None,
        dense_model_path: str | Path | None = None,
        retriever_mode: str = "hybrid",
        top_k: int = 3,
        candidate_k: int = 20,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        use_reranker: bool = False,
        reranker_model_path: str | Path | None = None,
        reranker_batch_size: int = 32,
        reranker_max_length: int = 512,
        max_snippet_chars: int = 500,
    ):
        project_root = Path(__file__).resolve().parents[2]
        if docs_path is None:
            docs_path = project_root / "data" / "processed" / "docs.jsonl"

        if index_dir is None:
            index_dir = project_root / "data" / "index"

        if dense_model_path is None:
            dense_model_path = project_root.parent / "models" / "Embedding" / "bge-base-en"

        index_dir = Path(index_dir)
        self.retriever_mode = retriever_mode
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.max_snippet_chars = max_snippet_chars
        self._cache: dict[tuple[str, int], list[SearchResults]] = {}

        if retriever_mode == "bm25":
            from search_agent.retriever.bm25 import BM25Retriever

            self.retriever = BM25Retriever(docs_path)
        elif retriever_mode == "dense":
            from search_agent.retriever.dense import DenseRetriever

            self.retriever = DenseRetriever(
                docs_path=docs_path,
                model_path=dense_model_path,
                index_path=index_dir / "faiss.index",
                meta_path=index_dir / "dense_docs.json",
            )
        elif retriever_mode == "hybrid":
            from search_agent.retriever.hybrid import HybridRetriever

            self.retriever = HybridRetriever(
                docs_path=docs_path,
                dense_model_path=dense_model_path,
                dense_index_path=index_dir / "faiss.index",
                dense_meta_path=index_dir / "dense_docs.json",
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                use_reranker=use_reranker,
                reranker_model_path=reranker_model_path,
                reranker_batch_size=reranker_batch_size,
                reranker_max_length=reranker_max_length,
            )
        else:
            raise ValueError(
                f"Unsupported retriever_mode: {retriever_mode}. "
                "Use one of: bm25, dense, hybrid."
            )

    def _truncate_snippet(self, text: str) -> str:
        if len(text) <= self.max_snippet_chars:
            return text
        return text[: self.max_snippet_chars].rstrip() + "..."

    def search(self, query: str, top_k: int | None = None) -> list[SearchResults]:
        if top_k is None:
            top_k = self.top_k

        cache_key = (query.strip().lower(), top_k)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self.retriever_mode == "hybrid":
            docs = self.retriever.search(
                query,
                top_k=top_k,
                candidate_k=self.candidate_k,
            )
        else:
            docs = self.retriever.search(query, top_k=top_k)

        results = [
            SearchResults(
                title=doc.title,
                url=f"local://{doc.doc_id}",
                snippet=self._truncate_snippet(doc.text),
                score=doc.score,
            )
            for doc in docs
        ]
        self._cache[cache_key] = results
        return results
