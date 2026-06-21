import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.memory.store import MemoryStore


def test_memory_store_can_save_and_load(tmp_path):
    memory_path = tmp_path / "agent_memory.jsonl"
    store = MemoryStore(memory_path=memory_path)

    result = {
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

    store.save_from_agent_result(
        question="Which city is the birthplace of the author of The Silent Harbor?",
        result=result,
    )

    items = store.load_all()
    assert len(items) == 1
    assert (
        items[0].question
        == "Which city is the birthplace of the author of The Silent Harbor?"
    )
    assert items[0].answer == "Brookhaven. [1]"
    assert items[0].search_queries == [
        "author of The Silent Harbor",
        "Lena Moris birthplace",
    ]


def test_memory_store_can_search_similar_question(tmp_path):
    memory_path = tmp_path / "agent_memory.jsonl"
    store = MemoryStore(memory_path=memory_path)

    result = {
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

    store.save_from_agent_result(
        question="Which city is the birthplace of the author of The Silent Harbor?",
        result=result,
    )

    memories = store.search(
        "Which city is the birthplace of the author of Glass Meridian?",
        top_k=1,
    )

    assert len(memories) == 1
    assert memories[0].search_queries == [
        "author of The Silent Harbor",
        "Lena Moris birthplace",
    ]


def test_memory_context_warns_not_to_use_memory_as_evidence(tmp_path):
    memory_path = tmp_path / "agent_memory.jsonl"
    store = MemoryStore(memory_path=memory_path)
    result = {
        "answer": "Brookhaven. [1]",
        "trace": [
            {
                "action": "search",
                "content": "author of The Silent Harbor",
            },
            {
                "action": "answer",
                "content": "Brookhaven. [1]",
                "accepted": True,
            },
        ],
    }

    store.save_from_agent_result(
        question="Which city is the birthplace of the author of The Silent Harbor?",
        result=result,
    )

    context = store.build_memory_context(
        "Which city is the birthplace of the author of Glass Meridian?"
    )

    assert "Relevant memory:" in context
    assert "Do not use them as evidence" in context
    assert "Final answers must still be supported by current <observe>" in context
