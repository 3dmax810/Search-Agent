import json
import re
from dataclasses import dataclass
from pathlib import Path
from rank_bm25 import BM25Okapi


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    source: str
    score: float


class BM25Retriever:
    def __init__(self, docs_path: str | Path):
        self.docs_path = Path(docs_path)
        self.docs = self._load_docs(self.docs_path)

        if not self.docs:
            raise ValueError(f"No documents found in {self.docs_path}")
        self.corpus_tokens = [
            self._tokenize(doc["title"] + " " + doc["text"]) for doc in self.docs
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _load_docs(self, path: Path) -> list[dict]:
        docs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[
            :top_k
        ]

        results = []
        for idx, score in ranked:
            doc = self.docs[idx]
            results.append(
                RetrievedDoc(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    text=doc["text"],
                    source=doc["source"],
                    score=float(score),
                )
            )
        return results
