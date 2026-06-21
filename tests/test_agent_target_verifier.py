import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.agent.runtime import AgentRuntime
from search_agent.tools.search_tool import SearchResults


class TargetVerifierModel:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)

        if self.calls == 1:
            return "<think>I need the county first.</think><search>Van Hook Township county</search>"
        if self.calls == 2:
            return "<answer>Mountrail County, North Dakota. [1]</answer>"
        if self.calls == 3:
            return "<think>I need the county seat.</think><search>Mountrail County, North Dakota county seat</search>"
        return "<answer>Stanley. [1]</answer>"


class TargetVerifierSearchTool:
    def search(self, query: str) -> list[SearchResults]:
        if "county seat" in query.lower():
            return [
                SearchResults(
                    title="Mountrail County, North Dakota",
                    url="mock://mountrail-county",
                    snippet="The county seat of Mountrail County is Stanley.",
                    score=1.0,
                )
            ]
        return [
            SearchResults(
                title="Van Hook Township",
                url="mock://van-hook-township",
                snippet="Van Hook Township is located in Mountrail County, North Dakota.",
                score=1.0,
            )
        ]


class AcceptingJudge:
    def judge(self, question: str, answer: str, observe_text: str):
        return True, "accepted in test"


def test_agent_target_verifier_rejects_mismatched_first_hop_answer():
    model = TargetVerifierModel()
    agent = AgentRuntime(
        model=model,
        search_tool=TargetVerifierSearchTool(),
        answer_judge=AcceptingJudge(),
        max_search_turns=5,
        max_model_turns=6,
        max_answer_tokens=8,
        use_answer_judge=True,
        use_memory=False,
        use_target_verifier=True,
    )

    result = agent.run(
        "What is the seat of the county where Van Hook Township is located?"
    )
    trace = result["trace"]
    target_rejections = [
        step
        for step in trace
        if step.get("state") == "ANSWER_REJECTED"
        and step.get("target_verifier_accepted") is False
    ]

    assert result["answer"] == "Stanley. [1]"
    assert len(target_rejections) == 1
    assert (
        target_rejections[0]["target_verifier_suggested_query"]
        == "Mountrail County, North Dakota county seat"
    )
    assert "Next action should be <search>Mountrail County, North Dakota county seat</search>" in trace[2]["prompt"]
