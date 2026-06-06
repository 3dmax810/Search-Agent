import time
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Protocol

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from search_agent.eval.baselines import BaselineResult
from search_agent.eval.metrics import exact_match, f1_score, citation_hit


class BaselineLike(Protocol):
    def run(self, question: str) -> BaselineResult: ...


def load_jsonl(path: str) -> list[dict]:
    rows = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _doc_id_from_url(url: str) -> str | None:
    if url.startswith("local://"):
        return url.replace("local://", "", 1)

    if url.startswith("mock://"):
        return url.replace("mock://", "", 1)
    return None


def resolve_cited_doc_ids(
    citations: list[str],
    retrieved_docs: list[dict],
) -> list[str]:
    cited_doc_ids = []
    for citation in citations:
        try:
            idx = int(citation) - 1
        except ValueError:
            continue

        if idx < 0 or idx >= len(retrieved_docs):
            continue

        url = retrieved_docs[idx].get("url", "")
        doc_id = _doc_id_from_url(url)
        if doc_id:
            cited_doc_ids.append(doc_id)

    return cited_doc_ids


def evaluate_baseline(
    baseline: BaselineLike,
    qa_rows: list[dict],
) -> list[dict]:
    eval_rows = []
    for qa in qa_rows:
        question = qa["question"]
        gold_answer = qa["answer"]
        supporting_doc_ids = qa.get("supporting_doc_ids", [])
        start_time = time.perf_counter()
        result = baseline.run(question=question)
        latency_seconds = time.perf_counter() - start_time
        cited_doc_ids = resolve_cited_doc_ids(
            result.citations,
            result.retrieved_docs,
        )

        row = {
            "method": result.method,
            "question": question,
            "gold_answer": gold_answer,
            "prediction": result.answer,
            "raw_output": result.raw_output,
            "citations": result.citations,
            "cited_doc_ids": cited_doc_ids,
            "retrieved_docs": result.retrieved_docs,
            "supporting_doc_ids": supporting_doc_ids,
            "em": exact_match(result.answer, gold_answer),
            "f1": f1_score(result.answer, gold_answer),
            "citation_hit": citation_hit(
                answer=result.answer,
                supporting_doc_ids=supporting_doc_ids,
                cited_doc_ids=cited_doc_ids,
            ),
            "search_turns": result.search_turns,
            "latency_seconds": latency_seconds,
            "format_error": float(not result.answer),
        }

        eval_rows.append(row)
    return eval_rows


def summarize_eval_rows(eval_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in eval_rows:
        grouped[row["method"]].append(row)
    summary = []
    for method, rows in grouped.items():
        count = len(rows)

        summary.append(
            {
                "method": method,
                "count": count,
                "em": sum(row["em"] for row in rows) / count,
                "f1": sum(row["f1"] for row in rows) / count,
                "citation_hit": sum(row["citation_hit"] for row in rows) / count,
                "avg_search_turns": sum(row["search_turns"] for row in rows) / count,
                "avg_latency_seconds": sum(row["latency_seconds"] for row in rows)
                / count,
                "format_error_rate": sum(row["format_error"] for row in rows) / count,
            }
        )

    return summary
