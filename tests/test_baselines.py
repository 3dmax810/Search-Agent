import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.eval.baselines import (
    NoSearchBaseline,
    SearchAgentBaseline,
    SingleShotRAGBaseline,
)
from search_agent.tools.search_tool import MockSearchTool


class FakeModel:
    def generate(self, prompt: str) -> str:
        if "Evidence:" in prompt:
            return "<answer>Brookhaven. [1]</answer>"
        return "<answer>Brookhaven</answer>"


def test_no_search_baseline_runs():
    baseline = NoSearchBaseline(model=FakeModel())
    result = baseline.run(
        "Which city is the birthplace of the author of The Silent Harbor?"
    )

    assert result.method == "no-search"
    assert result.answer == "Brookhaven"
    assert result.search_turns == 0


def test_single_shot_rag_baseline_runs():
    baseline = SingleShotRAGBaseline(model=FakeModel(), search_tool=MockSearchTool())

    result = baseline.run(
        "Which city is the birthplace of the author of The Silent Harbor?"
    )

    assert result.method == "single-shot-rag"
    assert result.answer == "Brookhaven. [1]"
    assert result.citations == ["1"]
    assert result.citations == ["1"]
    assert len(result.retrieved_docs) > 0


class FakeAgentWithTrace:
    def run(self, question: str) -> dict:
        return {
            "answer": "Brookhaven. [1]",
            "trace": [
                {
                    "action": "search",
                    "state": "SEARCH",
                    "search_results": [
                        {
                            "title": "Lena Moris",
                            "url": "local://doc002",
                            "snippet": "Lena Moris was born in Brookhaven.",
                            "score": 1.0,
                        }
                    ],
                },
                {"action": "search", "state": "DUPLICATE_SEARCH"},
                {
                    "action": "search",
                    "state": "HALT",
                    "halt_reason": "search limit reached",
                },
                {
                    "action": "answer",
                    "state": "ANSWER_REJECTED",
                    "answer_constraint_accepted": False,
                },
                {
                    "action": "answer",
                    "state": "ANSWER_REJECTED",
                    "target_verifier_accepted": False,
                },
                {"action": "answer", "state": "ANSWER_ACCEPTED"},
            ],
        }


def test_search_agent_baseline_counts_real_search_turns():
    baseline = SearchAgentBaseline(agent=FakeAgentWithTrace())

    result = baseline.run("Question?")

    assert result.search_turns == 1
    assert result.search_action_turns == 3
    assert result.model_turns == 6
    assert result.duplicate_search_count == 1
    assert result.search_limit_count == 1
    assert result.answer_reject_count == 2
    assert result.answer_constraint_reject_count == 1
    assert result.target_verifier_reject_count == 1
    assert result.final_state == "ANSWER_ACCEPTED"
