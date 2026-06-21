import argparse
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import json
from search_agent.agent.runtime import AgentRuntime
from search_agent.tools.search_tool import LocalSearchTool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--docs-path",
        default=None,
        help="Path to docs.jsonl. Defaults to data/processed/docs.jsonl.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Path to retrieval index directory. Defaults to data/index.",
    )
    parser.add_argument(
        "--embedding-model-path",
        default=None,
        help="Path to the sentence embedding model.",
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
        help="Maximum number of real search calls allowed.",
    )
    parser.add_argument(
        "--max-model-turns",
        type=int,
        default=8,
        help="Maximum number of model action turns.",
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

    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--use-memory",
        action="store_true",
        default=True,
        help="Enable episodic search-strategy memory. This is the default.",
    )
    memory_group.add_argument(
        "--no-memory",
        action="store_false",
        dest="use_memory",
        help="Disable memory retrieval and memory writing.",
    )

    args = parser.parse_args()

    if args.use_reranker and not args.reranker_model_path:
        parser.error("--reranker-model-path is required when --use-reranker is set.")

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
    agent = AgentRuntime(
        search_tool=search_tool,
        max_search_turns=args.max_search_turns,
        max_model_turns=args.max_model_turns,
        max_duplicate_searches=args.max_duplicate_searches,
        max_answer_tokens=args.max_answer_tokens or None,
        reject_intermediate_answers=not args.allow_intermediate_answers,
        use_target_verifier=args.use_target_verifier,
        use_answer_judge=args.use_answer_judge,
        use_memory=args.use_memory,
    )
    result = agent.run(args.question)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
