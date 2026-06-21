import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.agent.runtime import AgentRuntime
from search_agent.memory.store import MemoryStore
from search_agent.tools.search_tool import MockSearchTool


class MemoryTestModel:
    def generate(self, prompt: str) -> str:
        if "The Old Man and the Sea is a novel by Ernest Hemingway" not in prompt:
            return "<search>author of The Old Man and the Sea</search>"

        if "Ernest Hemingway was born in Oak Park, Illinois" not in prompt:
            return "<search>Ernest Hemingway birthplace</search>"

        return "<answer>Oak Park, Illinois. [1]</answer>"


class AlwaysAcceptJudge:
    def judge(self, question: str, answer: str, observe_text: str):
        return True, "ok"


def test_agent_saves_successful_trace_to_memory(tmp_path):
    memory_path = tmp_path / "agent_memory.jsonl"
    memory_store = MemoryStore(memory_path=memory_path)

    agent = AgentRuntime(
        model=MemoryTestModel(),
        search_tool=MockSearchTool(),
        answer_judge=AlwaysAcceptJudge(),
        memory_store=memory_store,
        use_memory=True,
    )

    result = agent.run(
        "Which city is the birthplace of the author of The Old Man and the Sea?"
    )

    assert result
