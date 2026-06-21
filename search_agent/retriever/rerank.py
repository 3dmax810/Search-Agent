from pathlib import Path

from search_agent.retriever.bm25 import RetrievedDoc


class CrossEncoderReranker:
    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder

        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = CrossEncoder(self.model_path, max_length=max_length)

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int = 5,
    ) -> list[RetrievedDoc]:
        if not docs:
            return []

        pairs = [(query, f"{doc.title}. {doc.text}") for doc in docs]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        reranked_docs = []
        for doc, score in zip(docs, scores):
            reranked_docs.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    source=doc.source,
                    score=float(score),
                )
            )

        reranked_docs.sort(key=lambda doc: doc.score, reverse=True)
        return reranked_docs[:top_k]
