import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(root_path)

from search_agent.agent.context import ContextAssembler


def test_context_assembler_builds_prompt_with_memory():
    assembler = ContextAssembler(max_context_chars=6000)
    context = assembler.assemble(
        question="Which city is the birthplace of the author of The Silent Harbor?",
        trace_text="<search>author of The Silent Harbor</search>",
        memory_context="Relevant memory:\nSearch strategy: search author, then birthplace.",
    )

    assert "Question:" in context.prompt
    assert (
        "Which city is the birthplace of the author of The Silent Harbor?"
        in context.prompt
    )
    assert "<search>author of The Silent Harbor</search>" in context.prompt
    assert "Relevant memory:" in context.prompt
    assert context.memory_used is True
    assert context.memory_chars > 0
    assert context.context_truncated is False


def test_context_assembler_without_memory():
    assembler = ContextAssembler(max_context_chars=6000)
    context = assembler.assemble(
        question="What is the hometown of the singer behind Midnight Kites?",
        trace_text="",
        memory_context="",
    )

    assert "What is the hometown of the singer behind Midnight Kites?" in context.prompt
    assert context.memory_used is False
    assert context.memory_chars == 0
    assert context.context_chars == 0
    assert context.context_truncated is False


def test_context_assembler_truncates_long_trace():
    assembler = ContextAssembler(max_context_chars=20)
    context = assembler.assemble(
        question="Question?",
        trace_text="0123456789abcdefghijklmnopqrstuvwxyz",
        memory_context="",
    )

    assert context.context_truncated is True
    assert context.trace_context.startswith("[context truncated]")
    assert context.trace_context.endswith("ghijklmnopqrstuvwxyz")
    assert context.context_chars == len(context.trace_context)
