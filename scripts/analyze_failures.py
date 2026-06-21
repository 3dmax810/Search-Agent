import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.eval.metrics import normalize_answer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def doc_id_from_url(url: str) -> str:
    if url.startswith("local://"):
        return url.replace("local://", "", 1)
    if url.startswith("mock://"):
        return url.replace("mock://", "", 1)
    return url


def parse_raw_trace(row: dict) -> list[dict]:
    raw_output = row.get("raw_output", "")
    if not raw_output:
        return []
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return []
    trace = parsed.get("trace", [])
    if isinstance(trace, list):
        return trace
    return []


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\[\d+\]", "", answer)
    answer = re.sub(r"</?answer>", "", answer)
    return " ".join(answer.split()).strip(" .")


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", normalize_answer(text)))


def collect_trace_docs(trace: list[dict]) -> tuple[list[dict], str]:
    docs = []
    observe_parts = []
    for step in trace:
        for doc in step.get("search_results", []) or []:
            docs.append(doc)
            observe_parts.append(
                " ".join(
                    str(doc.get(field, ""))
                    for field in ("title", "snippet", "url")
                    if doc.get(field)
                )
            )
    return docs, "\n".join(observe_parts)


def unique_doc_ids(docs: list[dict]) -> list[str]:
    seen = set()
    ids = []
    for doc in docs:
        doc_id = doc_id_from_url(str(doc.get("url", "")))
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
    return ids


def search_queries(trace: list[dict]) -> list[str]:
    queries = []
    for step in trace:
        if step.get("action") == "search" and step.get("content"):
            queries.append(str(step["content"]))
    return queries


def final_answer_contains_relation_text(question: str, answer: str) -> bool:
    question_norm = normalize_answer(question)
    answer_norm = normalize_answer(answer)

    # These words usually describe the bridge entity/relation rather than the short
    # final answer span. Keeping this heuristic conservative avoids over-classifying.
    relation_words = {
        "author",
        "actor",
        "performer",
        "singer",
        "founder",
        "founded",
        "distributed",
        "distributor",
        "company",
        "manufacturer",
        "owner",
        "located",
        "member",
        "spouse",
        "birthplace",
        "born",
        "written",
        "wrote",
    }
    question_relation_words = {
        word for word in relation_words if re.search(rf"\b{word}\b", question_norm)
    }
    answer_relation_words = {
        word for word in question_relation_words if re.search(rf"\b{word}\b", answer_norm)
    }

    return bool(answer_relation_words)


def is_intermediate_answer_suspect(question: str, prediction: str, gold_answer: str) -> bool:
    if not prediction or normalize_answer(prediction) == normalize_answer(gold_answer):
        return False

    answer = clean_answer(prediction)
    if token_count(answer) > 12:
        # Long explanatory answers are handled separately.
        return False

    answer_norm = normalize_answer(answer)
    question_norm = normalize_answer(question)

    bridge_patterns = [
        r"\bthe author\b",
        r"\bwas written by\b",
        r"\bwritten by\b",
        r"\bthe performer\b",
        r"\bperformed by\b",
        r"\bthe company\b",
        r"\bdistributed by\b",
        r"\bthe distributor\b",
        r"\bthe manufacturer\b",
        r"\bthe owner\b",
        r"\bmember of\b",
    ]
    if any(re.search(pattern, prediction.lower()) for pattern in bridge_patterns):
        return True

    # If the answer repeats bridge relation words from the question, it is often an
    # intermediate fact instead of the requested final value.
    if final_answer_contains_relation_text(question_norm, answer_norm):
        return True

    return False


def is_no_valid_answer_fallback(prediction: str) -> bool:
    prediction_norm = normalize_answer(prediction)
    fallback_phrases = [
        "i could not produce a valid answer",
        "within the model turn limit",
        "could not produce valid answer",
        "no valid answer",
    ]
    return any(phrase in prediction_norm for phrase in fallback_phrases)


def classify_row(
    row: dict,
    max_answer_tokens: int,
    duplicate_threshold: int,
) -> dict:
    trace = parse_raw_trace(row)
    trace_docs, observe_text = collect_trace_docs(trace)
    trace_doc_ids = unique_doc_ids(trace_docs)

    supporting_ids = set(row.get("supporting_doc_ids", []) or [])
    cited_ids = set(row.get("cited_doc_ids", []) or [])
    retrieved_ids = set(trace_doc_ids)

    retrieved_support_ids = sorted(retrieved_ids & supporting_ids)
    cited_support_ids = sorted(cited_ids & supporting_ids)

    support_count = len(supporting_ids)
    retrieved_support_recall = (
        len(retrieved_support_ids) / support_count if support_count else 0.0
    )
    citation_recall = len(cited_support_ids) / support_count if support_count else 0.0

    prediction = row.get("prediction", "") or ""
    gold_answer = row.get("gold_answer", "") or ""
    answer_tokens = token_count(prediction)
    gold_norm = normalize_answer(gold_answer)
    observe_norm = normalize_answer(observe_text)

    gold_in_observations = bool(gold_norm and gold_norm in observe_norm)
    no_valid_answer = is_no_valid_answer_fallback(prediction)
    long_answer = answer_tokens > max_answer_tokens
    intermediate_suspect = is_intermediate_answer_suspect(
        row.get("question", ""),
        prediction,
        gold_answer,
    )

    failure_signals = []
    if no_valid_answer:
        failure_signals.append("no_valid_answer")
    if long_answer:
        failure_signals.append("long_answer")
    if intermediate_suspect:
        failure_signals.append("intermediate_entity_answer")
    if row.get("duplicate_search_count", 0) >= duplicate_threshold:
        failure_signals.append("duplicate_query_loop")
    if row.get("answer_constraint_reject_count", 0) > 0:
        failure_signals.append("answer_constraint_reject")
    if row.get("target_verifier_reject_count", 0) > 0:
        failure_signals.append("target_verifier_reject")
    if row.get("format_error_count", 0) > 0:
        failure_signals.append("format_error")
    if row.get("search_limit_count", 0) > 0:
        failure_signals.append("search_limit")
    if gold_in_observations and row.get("em", 0.0) == 0.0:
        failure_signals.append("gold_observed_but_answer_wrong")

    if row.get("em", 0.0) == 1.0:
        failure_type = "correct"
    elif no_valid_answer:
        failure_type = "no_valid_answer_or_turn_limit"
    elif not retrieved_support_ids:
        failure_type = "retrieval_failure"
    elif long_answer:
        failure_type = "long_answer"
    elif intermediate_suspect:
        failure_type = "intermediate_entity_answer"
    elif gold_in_observations:
        failure_type = "answer_extraction_failure"
    elif retrieved_support_recall < 1.0:
        failure_type = "evidence_chain_incomplete"
    elif not cited_support_ids:
        failure_type = "citation_selection_failure"
    elif row.get("duplicate_search_count", 0) >= duplicate_threshold:
        failure_type = "query_planning_or_loop"
    elif row.get("final_state") != "ANSWER_ACCEPTED":
        failure_type = "halt_or_format_failure"
    else:
        failure_type = "unknown_failure"

    return {
        "method": row.get("method", ""),
        "question": row.get("question", ""),
        "gold_answer": gold_answer,
        "prediction": prediction,
        "failure_type": failure_type,
        "failure_signals": failure_signals,
        "em": row.get("em", 0.0),
        "f1": row.get("f1", 0.0),
        "citation_hit": row.get("citation_hit", 0.0),
        "supporting_doc_count": support_count,
        "retrieved_support_count": len(retrieved_support_ids),
        "retrieved_support_recall": retrieved_support_recall,
        "all_support_retrieved": float(bool(supporting_ids) and supporting_ids <= retrieved_ids),
        "cited_support_count": len(cited_support_ids),
        "citation_recall": citation_recall,
        "all_support_cited": float(bool(supporting_ids) and supporting_ids <= cited_ids),
        "gold_in_observations": float(gold_in_observations),
        "answer_token_count": answer_tokens,
        "no_valid_answer": float(no_valid_answer),
        "long_answer": float(long_answer),
        "intermediate_entity_answer_suspect": float(intermediate_suspect),
        "search_turns": row.get("search_turns", 0),
        "search_action_turns": row.get("search_action_turns", 0),
        "model_turns": row.get("model_turns", 0),
        "answer_reject_count": row.get("answer_reject_count", 0),
        "answer_constraint_reject_count": row.get(
            "answer_constraint_reject_count",
            0,
        ),
        "target_verifier_reject_count": row.get(
            "target_verifier_reject_count",
            0,
        ),
        "duplicate_search_count": row.get("duplicate_search_count", 0),
        "search_limit_count": row.get("search_limit_count", 0),
        "format_error_count": row.get("format_error_count", 0),
        "final_state": row.get("final_state", ""),
        "latency_seconds": row.get("latency_seconds", 0.0),
        "search_queries": search_queries(trace),
        "retrieved_support_ids": retrieved_support_ids,
        "cited_support_ids": cited_support_ids,
        "supporting_doc_ids": sorted(supporting_ids),
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["failure_type"]].append(row)

    summary = []
    total = len(rows)
    for failure_type, items in sorted(grouped.items()):
        count = len(items)
        summary.append(
            {
                "failure_type": failure_type,
                "count": count,
                "rate": count / total if total else 0.0,
                "avg_em": sum(item["em"] for item in items) / count,
                "avg_f1": sum(item["f1"] for item in items) / count,
                "avg_citation_hit": sum(item["citation_hit"] for item in items) / count,
                "avg_retrieved_support_recall": sum(
                    item["retrieved_support_recall"] for item in items
                )
                / count,
                "avg_citation_recall": sum(item["citation_recall"] for item in items)
                / count,
                "avg_duplicate_search_count": sum(
                    item["duplicate_search_count"] for item in items
                )
                / count,
                "avg_answer_constraint_reject_count": sum(
                    item["answer_constraint_reject_count"] for item in items
                )
                / count,
                "avg_target_verifier_reject_count": sum(
                    item["target_verifier_reject_count"] for item in items
                )
                / count,
                "avg_latency_seconds": sum(item["latency_seconds"] for item in items)
                / count,
            }
        )
    return summary


def write_markdown(path: Path, rows: list[dict], limit: int) -> None:
    lines = ["# Failure Analysis", ""]
    counts = Counter(row["failure_type"] for row in rows)
    lines.append("## Summary")
    for failure_type, count in counts.most_common():
        lines.append(f"- {failure_type}: {count}")
    lines.append("")

    lines.append(f"## Examples Top {min(limit, len(rows))}")
    for idx, row in enumerate(rows[:limit], 1):
        lines.extend(
            [
                "",
                f"### {idx}. {row['failure_type']}",
                f"- Question: {row['question']}",
                f"- Gold: {row['gold_answer']}",
                f"- Prediction: {row['prediction']}",
                f"- EM/F1/CitationHit: {row['em']} / {row['f1']} / {row['citation_hit']}",
                f"- Retrieved support recall: {row['retrieved_support_recall']}",
                f"- Citation recall: {row['citation_recall']}",
                f"- Signals: {', '.join(row['failure_signals']) or 'none'}",
                f"- Search queries: {' -> '.join(row['search_queries'])}",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--details-path",
        default=str(root_path / "results" / "eval_details.jsonl"),
        help="Path to eval_details.jsonl generated by scripts/run_eval.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the directory containing details-path.",
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Include EM=1 rows. By default only failed rows are written.",
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=8,
        help="Answers longer than this are flagged as long_answer.",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=int,
        default=2,
        help="Duplicate searches at or above this value are flagged as a loop signal.",
    )
    parser.add_argument(
        "--markdown-limit",
        type=int,
        default=20,
        help="Number of examples written to failure_report.md.",
    )
    args = parser.parse_args()

    details_path = Path(args.details_path)
    output_dir = Path(args.output_dir) if args.output_dir else details_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(details_path)
    analyzed = [
        classify_row(
            row,
            max_answer_tokens=args.max_answer_tokens,
            duplicate_threshold=args.duplicate_threshold,
        )
        for row in rows
        if args.include_correct or row.get("em", 0.0) < 1.0
    ]
    summary = summarize(analyzed)

    details_out = output_dir / "failure_analysis.jsonl"
    summary_out = output_dir / "failure_summary.csv"
    report_out = output_dir / "failure_report.md"

    write_jsonl(details_out, analyzed)
    write_csv(summary_out, summary)
    write_markdown(report_out, analyzed, limit=args.markdown_limit)

    print(f"Wrote failure details: {details_out}")
    print(f"Wrote failure summary: {summary_out}")
    print(f"Wrote failure report: {report_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
