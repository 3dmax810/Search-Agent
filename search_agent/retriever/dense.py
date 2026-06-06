import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from search_agent.retriever.bm25 import RetrievedDoc


class DenseRetriever:
    def __init__(
        self,
        docs_path: str | Path,
        model_path: str | Path,
        index_path: str | Path,
        meta_path: str | Path,
    ):
        self.docs_path = Path(docs_path)
        self.model_path = str(model_path)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)

        self.model = SentenceTransformer(self.model_path)
        self.docs = self._load_docs(self.docs_path)
        self.index = None

        if self.index_path.exists() and self.meta_path.exists():
            self.load_index()

    def _load_docs(self, path: Path) -> list[dict]:
        docs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs

    def _doc_texts(self) -> list[str]:
        return [f"{doc['title']}. {doc['text']}" for doc in self.docs]

    def build_index(self, batch_size: int = 32) -> None:
        if not self.docs:
            raise ValueError(f"No documents found in {self.docs_path}")

        texts = self._doc_texts()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)

        self.index = index

    def load_index(self) -> None:
        self.index = faiss.read_index(str(self.index_path))

        with self.meta_path.open("r", encoding="utf-8") as f:
            self.docs = json.load(f)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        if self.index is None:
            raise ValueError("Dense index is not loaded. Run build_index() first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            doc = self.docs[int(idx)]
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
