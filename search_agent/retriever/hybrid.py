import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from search_agent.retriever.bm25 import BM25Retriever, RetrievedDoc
from search_agent.retriever.dense import DenseRetriever
from search_agent.retriever.rerank import CrossEncoderReranker


class HybridRetriever:
    def __init__(
        self,
        docs_path: str | Path,
        dense_model_path: str | Path,
        dense_index_path: str | Path,
        dense_meta_path: str | Path,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        use_reranker: bool = False,
        reranker_model_path: str | Path | None = None,
        reranker_batch_size: int = 32,
        reranker_max_length: int = 512,
    ):
        self.bm25 = BM25Retriever(docs_path)
        self.dense = DenseRetriever(
            docs_path=docs_path,
            model_path=dense_model_path,
            index_path=dense_index_path,
            meta_path=dense_meta_path,
        )
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.reranker = None

        if use_reranker:
            if reranker_model_path is None:
                raise ValueError(
                    "reranker_model_path is required when use_reranker=True"
                )
            self.reranker = CrossEncoderReranker(
                model_path=reranker_model_path,
                batch_size=reranker_batch_size,
                max_length=reranker_max_length,
            )

    def _normalize_scores(self, docs: list[RetrievedDoc]) -> dict[str, float]:
        if not docs:
            return {}
        scores = [doc.score for doc in docs]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return {doc.doc_id: 1.0 for doc in docs}

        return {
            doc.doc_id: (doc.score - min_score) / (max_score - min_score)
            for doc in docs
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[RetrievedDoc]:
        bm25_docs = self.bm25.search(query, top_k=candidate_k)
        dense_docs = self.dense.search(query, top_k=candidate_k)

        bm25_scores = self._normalize_scores(bm25_docs)
        dense_scores = self._normalize_scores(dense_docs)

        doc_map: dict[str, RetrievedDoc] = {}

        for doc in bm25_docs + dense_docs:
            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = doc

        merged = []

        for doc_id, doc in doc_map.items():
            score = self.bm25_weight * bm25_scores.get(
                doc_id, 0.0
            ) + self.dense_weight * dense_scores.get(doc_id, 0.0)

            merged.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    title=doc.title,
                    text=doc.text,
                    source=doc.source,
                    score=score,
                )
            )

        merged.sort(key=lambda doc: doc.score, reverse=True)

        if self.reranker is not None:
            return self.reranker.rerank(query, merged, top_k=top_k)
        return merged[:top_k]
