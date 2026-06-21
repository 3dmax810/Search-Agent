import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.agent.runtime import AgentRuntime
from search_agent.tools.search_tool import MockSearchTool


class RepeatingSearchModel:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)

        if self.calls == 1:
            return "<search>author of The Old Man and the Sea</search>"
        if self.calls == 2:
            return "<search>author of The Old Man and the Sea</search>"
        if self.calls == 3:
            return "<search>Ernest Hemingway birthplace</search>"
        return "<answer>Ernest Hemingway was born in Oak Park, Illinois. [1]</answer>"


class AcceptingJudge:
    def judge(self, question: str, answer: str, observe_text: str):
        return True, "accepted in test"


def test_agent_rejects_duplicate_search_and_shows_searched_queries():
    model = RepeatingSearchModel()
    agent = AgentRuntime(
        model=model,
        search_tool=MockSearchTool(),
        answer_judge=AcceptingJudge(),
        max_search_turns=3,
        max_model_turns=6,
        max_duplicate_searches=1,
        use_answer_judge=True,
        use_memory=False,
    )

    result = agent.run(
        "Which city is the birthplace of the author of The Old Man and the Sea?"
    )
    trace = result["trace"]

    duplicate_steps = [
        step for step in trace if step.get("state") == "DUPLICATE_SEARCH"
    ]
    real_search_steps = [
        step
        for step in trace
        if step.get("action") == "search"
        and step.get("state") == "SEARCH"
        and step.get("search_results")
    ]

    assert result["answer"] == "Ernest Hemingway was born in Oak Park, Illinois. [1]"
    assert len(duplicate_steps) == 1
    assert duplicate_steps[0]["duplicate_reason"] == "same normalized query"
    assert len(real_search_steps) == 2
    assert "Search control:" in model.prompts[1]
    assert "author of The Old Man and the Sea" in model.prompts[1]
