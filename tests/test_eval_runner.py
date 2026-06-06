import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.eval.baselines import BaselineResult
from search_agent.eval.runners import (
    evaluate_baseline,
    resolve_cited_doc_ids,
    summarize_eval_rows,
)


class FakeBaseline:
    def run(self, question: str) -> BaselineResult:
        return BaselineResult(
            method="fake",
            question=question,
            answer="Brookhaven [1]",
            raw_output="<answer>Brookhaven [1]</answer>",
            citations=["1"],
            search_turns=1,
            retrieved_docs=[
                {
                    "title": "Lena Moris",
                    "url": "local://doc002",
                    "snippet": "Lena Moris was born in Brookhaven",
                    "score": 1.0,
                }
            ],
        )


def test_resolve_cited_doc_ids():
    cited = resolve_cited_doc_ids(
        citations=["1"],
        retrieved_docs=[
            {
                "title": "Lena Moris",
                "url": "local://doc002",
                "snippet": "Lena Moris was born in Brookhaven",
                "score": 1.0,
            }
        ],
    )

    assert cited == ["doc002"]


def test_evaluate_baseline():
    qa_rows = [
        {
            "question": "Which city is the birthplace of the author of The Silent Harbor?",
            "answer": "Brookhaven",
            "supporting_doc_ids": ["doc001", "doc002"],
            "type": "multi-hop",
        }
    ]

    rows = evaluate_baseline(FakeBaseline(), qa_rows)

    assert len(rows) == 1
    assert rows[0]["method"] == "fake"
    assert rows[0]["em"] == 1.0
    assert rows[0]["citation_hit"] == 1.0
    assert rows[0]["search_turns"] == 1


def test_summarize_eval_rows():
    rows = [
        {
            "method": "fake",
            "em": 1.0,
            "f1": 1.0,
            "citation_hit": 1.0,
            "search_turns": 1,
            "latency_seconds": 1.0,
            "format_error": 0.0,
        },
        {
            "method": "fake",
            "em": 0.0,
            "f1": 0.5,
            "citation_hit": 0.0,
            "search_turns": 2,
            "latency_seconds": 1.0,
            "format_error": 0.0,
        },
    ]

    summary = summarize_eval_rows(rows)
    assert len(summary) == 1
    assert summary[0]["method"] == "fake"
    assert summary[0]["count"] == 2
    assert summary[0]["em"] == 0.5
    assert summary[0]["f1"] == 0.75
    assert summary[0]["avg_latency_seconds"] == 1.0
    assert summary[0]["format_error_rate"] == 0.0
