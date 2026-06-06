import argparse
import json
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.retriever.bm25 import BM25Retriever
from search_agent.retriever.dense import DenseRetriever


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

    return rows


def validate_docs(docs: list[dict]) -> None:
    required_fields = {"doc_id", "title", "text", "source"}
    seen_ids = set()

    for idx, doc in enumerate(docs, 1):
        missing = required_fields - set(doc)
        if missing:
            raise ValueError(f"Doc line {idx} missing fields: {sorted(missing)}")

        doc_id = doc["doc_id"]
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate doc_id: {doc_id}")

        seen_ids.add(doc_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-model-path",
        default=str(root_path.parent / "models" / "Embedding" / "bge-base-zh"),
    )
    parser.add_argument(
        "--build-dense", action="store_true", help="Build FAISS dense retrieval index."
    )
    parser.add_argument(
        "--docs-path",
        default=str(root_path / "data" / "processed" / "docs.jsonl"),
    )
    parser.add_argument(
        "--index-dir",
        default=str(root_path / "data" / "index"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    docs_path = Path(args.docs_path)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs file not found: {docs_path}")

    docs = load_jsonl(docs_path)
    validate_docs(docs)

    retriever = BM25Retriever(docs_path)

    smoke_queries = [
        "author of The Silent Harbor",
        "Lena Moris birthplace",
        "Blue Orbit actor birthplace",
    ]

    smoke_results = {}

    for query in smoke_queries:
        results = retriever.search(query, top_k=args.top_k)
        smoke_results[query] = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "score": doc.score,
            }
            for doc in results
        ]

    meta = {
        "retriever": "bm25",
        "docs_path": str(docs_path),
        "doc_count": len(docs),
        "top_k": args.top_k,
        "smoke_results": smoke_results,
    }
    meta_path = index_dir / "bm25_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.build_dense:
        dense_index_path = index_dir / "faiss.index"
        dense_meta_path = index_dir / "dense_docs.json"

        dense_retriever = DenseRetriever(
            docs_path=docs_path,
            model_path=args.embedding_model_path,
            index_path=dense_index_path,
            meta_path=dense_meta_path,
        )
        dense_retriever.build_index()

        dense_smoke_results = {}
        for query in smoke_queries:
            results = dense_retriever.search(query, top_k=args.top_k)
            dense_smoke_results[query] = [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "score": doc.score,
                }
                for doc in results
            ]

        dense_report = {
            "retriever": "dense",
            "embedding_model_path": args.embedding_model_path,
            "index_path": str(dense_index_path),
            "meta_path": str(dense_meta_path),
            "doc_count": len(docs),
            "top_k": args.top_k,
            "smoke_results": dense_smoke_results,
        }

        dense_report_path = index_dir / "dense_meta.json"
        with dense_report_path.open("w", encoding="utf-8") as f:
            json.dump(dense_report, f, ensure_ascii=False, indent=2)

        meta["dense"] = dense_report

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
