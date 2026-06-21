import sys
import types

from search_agent.retriever.bm25 import RetrievedDoc
from search_agent.retriever.rerank import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, model_path, max_length=512):
        self.model_path = model_path
        self.max_length = max_length

    def predict(
        self,
        pairs,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        scores = []
        for _, passage in pairs:
            if "best evidence" in passage:
                scores.append(3.0)
            elif "medium evidence" in passage:
                scores.append(2.0)
            else:
                scores.append(1.0)
        return scores


def test_cross_encoder_reranker_reorders_docs(monkeypatch):
    fake_module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = CrossEncoderReranker(model_path="fake-reranker")
    docs = [
        RetrievedDoc("doc1", "Low", "weak evidence", "test", 0.9),
        RetrievedDoc("doc2", "Best", "best evidence", "test", 0.1),
        RetrievedDoc("doc3", "Medium", "medium evidence", "test", 0.5),
    ]

    results = reranker.rerank("query", docs, top_k=2)

    assert [doc.doc_id for doc in results] == ["doc2", "doc3"]
    assert [doc.score for doc in results] == [3.0, 2.0]
