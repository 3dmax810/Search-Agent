import argparse
import csv
import json
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.eval.baselines import (
    NoSearchBaseline,
    SingleShotRAGBaseline,
    SearchAgentBaseline,
)
from search_agent.agent.runtime import AgentRuntime
from search_agent.eval.runners import (
    evaluate_baseline,
    load_jsonl,
    summarize_eval_rows,
)
from search_agent.tools.search_tool import LocalSearchTool


def render_progress(
    sample_index: int,
    total: int,
    method: str,
    stage: str,
    width: int = 28,
) -> None:
    ratio = sample_index / total if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r[{bar}] {sample_index}/{total} method={method} stage={stage}",
        end="",
        flush=True,
    )
    if sample_index == total and stage == "done":
        print()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-path",
        default=str(root_path / "data" / "eval" / "qa.jsonl"),
    )
    parser.add_argument(
        "--results-dir",
        default=str(root_path / "results"),
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
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved documents per search turn.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of BM25/dense candidates used before hybrid fusion.",
    )
    parser.add_argument(
        "--retriever-mode",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
        help="Retriever backend used by the search tool.",
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
        "--max-snippet-chars",
        type=int,
        default=500,
        help="Maximum characters kept from each retrieved paragraph.",
    )
    parser.add_argument(
        "--max-search-turns",
        type=int,
        default=3,
        help="Maximum number of real search calls allowed for Search-Agent.",
    )
    parser.add_argument(
        "--max-model-turns",
        type=int,
        default=8,
        help="Maximum number of model action turns for Search-Agent.",
    )
    parser.add_argument(
        "--max-duplicate-searches",
        type=int,
        default=2,
        help="Maximum duplicate search attempts before forcing the model to stop repeating queries.",
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=8,
        help="Maximum tokens allowed in a final answer span. Use 0 to disable.",
    )
    parser.add_argument(
        "--allow-intermediate-answers",
        action="store_true",
        help="Disable the runtime guard that rejects likely intermediate-entity answers.",
    )
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--use-target-verifier",
        action="store_true",
        default=True,
        help="Enable target-aware answer verification. This is the default.",
    )
    target_group.add_argument(
        "--no-target-verifier",
        action="store_false",
        dest="use_target_verifier",
        help="Disable target-aware answer verification for ablation.",
    )
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--use-answer-judge",
        action="store_true",
        default=True,
        help="Enable LLM answer judge. This is the default.",
    )
    judge_group.add_argument(
        "--no-answer-judge",
        action="store_false",
        dest="use_answer_judge",
        help="Disable LLM answer judge to reduce latency.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Use a small sample first because LLM evaluation is slow.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["no-search", "single-shot-rag", "search-agent"],
        choices=["no-search", "single-shot-rag", "search-agent"],
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-sample progress bar output.",
    )
    args = parser.parse_args()

    if args.use_reranker and not args.reranker_model_path:
        parser.error("--reranker-model-path is required when --use-reranker is set.")

    qa_rows = load_jsonl(args.qa_path)
    if args.sample_size:
        qa_rows = qa_rows[: args.sample_size]

    search_tool = LocalSearchTool(
        docs_path=args.docs_path,
        index_dir=args.index_dir,
        dense_model_path=args.embedding_model_path,
        retriever_mode=args.retriever_mode,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        use_reranker=args.use_reranker,
        reranker_model_path=args.reranker_model_path,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        max_snippet_chars=args.max_snippet_chars,
    )

    baselines = []

    if "no-search" in args.methods:
        baselines.append(NoSearchBaseline())
    if "single-shot-rag" in args.methods:
        baselines.append(SingleShotRAGBaseline(search_tool=search_tool, top_k=args.top_k))
    if "search-agent" in args.methods:
        agent = AgentRuntime(
            search_tool=search_tool,
            max_search_turns=args.max_search_turns,
            max_model_turns=args.max_model_turns,
            max_duplicate_searches=args.max_duplicate_searches,
            max_answer_tokens=args.max_answer_tokens or None,
            reject_intermediate_answers=not args.allow_intermediate_answers,
            use_target_verifier=args.use_target_verifier,
            use_answer_judge=args.use_answer_judge,
        )
        baselines.append(SearchAgentBaseline(agent=agent))

    all_eval_rows = []

    for baseline in baselines:
        print(f"Running {baseline.__class__.__name__} on {len(qa_rows)} samples...")
        rows = evaluate_baseline(
            baseline,
            qa_rows,
            progress_callback=None if args.no_progress else render_progress,
        )
        all_eval_rows.extend(rows)

    summary_rows = summarize_eval_rows(all_eval_rows)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    details_path = results_dir / "eval_details.jsonl"
    summary_path = results_dir / "eval_summary.csv"

    write_jsonl(details_path, all_eval_rows)
    write_csv(summary_path, summary_rows)

    print(f"Wrote details: {details_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
