import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.eval.baselines import NoSearchBaseline, SingleShotRAGBaseline


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
    baseline = SingleShotRAGBaseline(model=FakeModel())

    result = baseline.run(
        "Which city is the birthplace of the author of The Silent Harbor?"
    )

    assert result.method == "single-shot-rag"
    assert result.answer == "Brookhaven. [1]"
    assert result.citations == ["1"]
    assert result.citations == ["1"]
    assert len(result.retrieved_docs) > 0
