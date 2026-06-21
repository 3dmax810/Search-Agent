import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.agent.runtime import AgentRuntime
from search_agent.tools.search_tool import MockSearchTool

class FakeModelForState:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str)->str:
        self.calls += 1

        if self.calls == 1:
            return "<search>author of The Old Man and the Sea</search>"

        if self.calls == 2:
            return "<search>Ernest Hemingway birthplace</search>" 

        return "<answer>Ernest Hemingway was born in Oak Park, Illinois. [1]</answer>"

class FakeAnswerJudge:
    def judge(self, question: str, answer: str, observe_text: str):
        return True, "accepted in test"


def test_agent_trace_records_runtime_states():
    agent = AgentRuntime(
        model=FakeModelForState(),
        search_tool=MockSearchTool(),
        answer_judge=FakeAnswerJudge(),
        use_answer_judge=True,
        use_memory=False,
    )

    result = agent.run(
        "Which city is the birthplace of the author of The Old Man and the Sea?"
    )

    assert result["answer"] == "Ernest Hemingway was born in Oak Park, Illinois. [1]"

    trace = result["trace"]
    assert [step["action"] for step in trace] == ["search", "search", "answer"]
    assert [step["state"] for step in trace] == [
        "SEARCH",
        "SEARCH",
        "ANSWER_ACCEPTED",
    ]
    assert trace[0]["halt_reason"] == ""
    assert trace[1]["halt_reason"] == ""
    assert trace[2]["accepted"] is True


def test_agent_trace_records_search_progress():
    agent = AgentRuntime(
        model=FakeModelForState(),
        search_tool=MockSearchTool(),
        answer_judge=FakeAnswerJudge(),
        use_answer_judge=True,
        use_memory=False,
    )

    result = agent.run(
        "Which city is the birthplace of the author of The Old Man and the Sea?"
    )

    first_search, second_search, final_answer = result["trace"]
    assert first_search["search_turns"] == 1
    assert second_search["search_turns"] == 2
    assert final_answer["search_turns"] == 2
    assert second_search["searched_queries"] == [
        "author of The Old Man and the Sea",
        "Ernest Hemingway birthplace",
    ]
