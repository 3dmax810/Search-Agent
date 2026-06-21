import argparse
import csv
import json
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

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


def parse_top_ks(value: str) -> list[int]:
    top_ks = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        top_k = int(part)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive: {top_k}")
        top_ks.append(top_k)

    if not top_ks:
        raise ValueError("At least one top_k value is required.")
    return sorted(set(top_ks))


def make_retriever(args):
    docs_path = Path(args.docs_path)
    index_dir = Path(args.index_dir)

    if args.retriever == "bm25":
        from search_agent.retriever.bm25 import BM25Retriever

        return BM25Retriever(docs_path)

    if args.retriever == "dense":
        from search_agent.retriever.dense import DenseRetriever

        return DenseRetriever(
            docs_path=docs_path,
            model_path=args.embedding_model_path,
            index_path=index_dir / "faiss.index",
            meta_path=index_dir / "dense_docs.json",
        )

    if args.retriever == "hybrid":
        from search_agent.retriever.hybrid import HybridRetriever

        return HybridRetriever(
            docs_path=docs_path,
            dense_model_path=args.embedding_model_path,
            dense_index_path=index_dir / "faiss.index",
            dense_meta_path=index_dir / "dense_docs.json",
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            use_reranker=args.use_reranker,
            reranker_model_path=args.reranker_model_path,
            reranker_batch_size=args.reranker_batch_size,
            reranker_max_length=args.reranker_max_length,
        )

    raise ValueError(f"Unsupported retriever: {args.retriever}")


def retrieval_metrics(gold_ids: set[str], retrieved_ids: list[str], top_k: int) -> dict:
    top_ids = retrieved_ids[:top_k]
    top_id_set = set(top_ids)
    hits = gold_ids & top_id_set

    first_hit_rank = 0
    for rank, doc_id in enumerate(top_ids, 1):
        if doc_id in gold_ids:
            first_hit_rank = rank
            break

    return {
        f"hit@{top_k}": 1.0 if hits else 0.0,
        f"all_support_hit@{top_k}": 1.0 if gold_ids <= top_id_set else 0.0,
        f"support_recall@{top_k}": len(hits) / len(gold_ids) if gold_ids else 0.0,
        f"mrr@{top_k}": 1.0 / first_hit_rank if first_hit_rank else 0.0,
    }


def search_retriever(
    retriever,
    query: str,
    top_k: int,
    candidate_k: int,
    retriever_name: str,
):
    if retriever_name == "hybrid":
        return retriever.search(query, top_k=top_k, candidate_k=candidate_k)
    return retriever.search(query, top_k=top_k)


def evaluate_retriever(
    retriever,
    qa_rows: list[dict],
    top_ks: list[int],
    candidate_k: int,
    retriever_name: str,
) -> list[dict]:
    max_top_k = max(top_ks)
    rows = []

    for qa in qa_rows:
        question = qa["question"]
        gold_ids = set(qa.get("supporting_doc_ids", []))
        results = search_retriever(
            retriever,
            query=question,
            top_k=max_top_k,
            candidate_k=candidate_k,
            retriever_name=retriever_name,
        )
        retrieved_ids = [doc.doc_id for doc in results]

        row = {
            "question": question,
            "answer": qa.get("answer", ""),
            "source_id": qa.get("source_id", ""),
            "hop_count": qa.get("hop_count", ""),
            "gold_doc_count": len(gold_ids),
            "retrieved_doc_ids": retrieved_ids,
            "supporting_doc_ids": sorted(gold_ids),
        }

        for top_k in top_ks:
            row.update(retrieval_metrics(gold_ids, retrieved_ids, top_k))

        rows.append(row)

    return rows


def summarize(rows: list[dict], top_ks: list[int], args) -> dict:
    summary = {
        "retriever": args.retriever,
        "count": len(rows),
        "candidate_k": args.candidate_k if args.retriever == "hybrid" else "",
        "bm25_weight": args.bm25_weight if args.retriever == "hybrid" else "",
        "dense_weight": args.dense_weight if args.retriever == "hybrid" else "",
        "use_reranker": args.use_reranker if args.retriever == "hybrid" else "",
        "reranker_model_path": args.reranker_model_path
        if args.retriever == "hybrid" and args.use_reranker
        else "",
    }

    if not rows:
        return summary

    for top_k in top_ks:
        for metric in [
            f"hit@{top_k}",
            f"all_support_hit@{top_k}",
            f"support_recall@{top_k}",
            f"mrr@{top_k}",
        ]:
            summary[metric] = sum(row[metric] for row in rows) / len(rows)

    return summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-path",
        default=str(root_path / "data" / "eval" / "qa.jsonl"),
    )
    parser.add_argument(
        "--docs-path",
        default=str(root_path / "data" / "processed" / "docs.jsonl"),
    )
    parser.add_argument(
        "--index-dir",
        default=str(root_path / "data" / "index"),
    )
    parser.add_argument(
        "--embedding-model-path",
        default=str(root_path.parent / "models" / "Embedding" / "bge-base-en"),
    )
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
    )
    parser.add_argument(
        "--top-ks",
        default="1,3,5,10",
        help="Comma-separated top-k values, for example: 1,3,5,10",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Hybrid candidate pool size before final top-k selection.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="BM25 score weight for hybrid retrieval.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="Dense score weight for hybrid retrieval.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use a CrossEncoder reranker after hybrid candidate fusion.",
    )
    parser.add_argument(
        "--reranker-model-path",
        default=None,
        help="Path or model name for a CrossEncoder reranker.",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=32,
        help="Batch size for CrossEncoder reranking.",
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
        help="Maximum sequence length for CrossEncoder reranking.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="0 means evaluate all rows.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_path / "results" / "retrieval"),
    )
    args = parser.parse_args()

    if args.use_reranker and not args.reranker_model_path:
        parser.error("--reranker-model-path is required when --use-reranker is set.")

    top_ks = parse_top_ks(args.top_ks)
    qa_rows = load_jsonl(Path(args.qa_path))
    if args.sample_size:
        qa_rows = qa_rows[: args.sample_size]

    retriever = make_retriever(args)
    rows = evaluate_retriever(
        retriever,
        qa_rows,
        top_ks,
        candidate_k=args.candidate_k,
        retriever_name=args.retriever,
    )
    summary = summarize(rows, top_ks, args)

    output_dir = Path(args.output_dir)
    output_prefix = args.retriever
    if args.retriever == "hybrid" and args.use_reranker:
        output_prefix = "hybrid_reranker"

    details_path = output_dir / f"{output_prefix}_retrieval_details.jsonl"
    summary_path = output_dir / f"{output_prefix}_retrieval_summary.csv"

    write_jsonl(details_path, rows)
    write_csv(summary_path, summary)

    print(f"Wrote details: {details_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
