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
from search_agent.eval.runners import (
    evaluate_baseline,
    load_jsonl,
    summarize_eval_rows,
)


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
    args = parser.parse_args()
    qa_rows = load_jsonl(args.qa_path)
    if args.sample_size:
        qa_rows = qa_rows[: args.sample_size]

    baselines = []

    if "no-search" in args.methods:
        baselines.append(NoSearchBaseline())
    if "single-shot-rag" in args.methods:
        baselines.append(SingleShotRAGBaseline())
    if "search-agent" in args.methods:
        baselines.append(SearchAgentBaseline())

    all_eval_rows = []

    for baseline in baselines:
        print(f"Running {baseline.__class__.__name__} on {len(qa_rows)} samples...")
        rows = evaluate_baseline(baseline, qa_rows)
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
