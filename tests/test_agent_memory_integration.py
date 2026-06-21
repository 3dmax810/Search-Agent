from openai.types.beta.realtime import response_audio_delta_event
from openai.types.beta.realtime import response_audio_delta_event
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.agent.runtime import AgentRuntime
from search_agent.memory.store import MemoryStore
from search_agent.tools.search_tool import MockSearchTool


class FakeModelForMemory:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1

        if self.calls == 1:
            return "<search>author of The Old Man and the Sea</search>"
        if self.calls == 2:
            return "<search>Ernest Hemingway birthplace</search>"
        return "<answer>Ernest Hemingway was born in Oak Park, Illinois. [1]</answer>"


class FakeAnswerJudge:
    def judge(self, question: str, answer: str, observe_text: str):
        return True, "accepted in test"


def test_agent_saves_successful_trace_to_memory(tmp_path):
    memory_path = tmp_path / "agent_memory.jsonl"
    memory_store = MemoryStore(memory_path=memory_path)

    agent = AgentRuntime(
        model=FakeModelForMemory(),
        search_tool=MockSearchTool(),
        answer_judge=FakeAnswerJudge(),
        memory_store=memory_store,
        use_answer_judge=True,
        use_memory=True,
    )

    result = agent.run(
        "Which city is the birthplace of the author of The Old Man and the Sea?"
    )

    assert result["answer"] == "Ernest Hemingway was born in Oak Park, Illinois. [1]"
    memories = memory_store.load_all()
    assert len(memories) == 1
    assert (
        memories[0].question
        == "Which city is the birthplace of the author of The Old Man and the Sea?"
    )
    assert memories[0].answer == "Ernest Hemingway was born in Oak Park, Illinois. [1]"
    assert memories[0].search_queries == [
        "author of The Old Man and the Sea",
        "Ernest Hemingway birthplace",
    ]


def test_agent_can_disable_memory(tmp_path):
    memory_path = tmp_path / "agent_memory.jsonl"
    memory_store = MemoryStore(memory_path=memory_path)

    agent = AgentRuntime(
        model=FakeModelForMemory(),
        search_tool=MockSearchTool(),
        answer_judge=FakeAnswerJudge(),
        memory_store=memory_store,
        use_answer_judge=True,
        use_memory=False,
    )

    result = agent.run(
        "Which city is the birthplace of the author of The Old Man and the Sea?"
    )

    assert result["answer"] == "Ernest Hemingway was born in Oak Park, Illinois. [1]"
    assert memory_store.load_all() == []


def test_agent_injects_memory_context_into_prompt(tmp_path):
    memory_path = tmp_path / "agent_memory.jsonl"
    memory_store = MemoryStore(memory_path=memory_path)

    seed_result = {
        "answer": "Brookhaven. [1]",
        "trace": [
            {
                "action": "search",
                "content": "author of The Silent Harbor",
            },
            {
                "action": "search",
                "content": "Lena Moris birthplace",
            },
            {
                "action": "answer",
                "content": "Brookhaven. [1]",
                "accepted": True,
            },
        ],
    }

    memory_store.save_from_agent_result(
        question="Which city is the birthplace of the author of The Silent Harbor?",
        result=seed_result,
    )

    agent = AgentRuntime(
        model=FakeModelForMemory(),
        search_tool=MockSearchTool(),
        answer_judge=FakeAnswerJudge(),
        memory_store=memory_store,
        use_answer_judge=True,
        use_memory=True,
    )

    result = agent.run(
        "Which city is the birthplace of the author of The Old Man and the Sea?",
    )
    first_prompt = result["trace"][0]["prompt"]

    assert "Relevant memory:" in first_prompt
    assert (
        "Past question: Which city is the birthplace of the author of The Silent Harbor?"
        in first_prompt
    )
    assert (
        "Search strategy: author of The Silent Harbor -> Lena Moris birthplace"
        in first_prompt
    )
    assert result["trace"][0]["memory_used"] is True
    assert result["trace"][0]["memory_chars"] > 0
