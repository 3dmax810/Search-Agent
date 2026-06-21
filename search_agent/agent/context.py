from dataclasses import dataclass

import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from search_agent.agent.prompts import build_prompt


@dataclass
class AssembledContext:
    prompt: str
    trace_context: str
    memory_context: str
    context_chars: int
    context_truncated: bool
    memory_chars: int
    memory_used: bool


class ContextAssembler:
    def __init__(self, max_context_chars: int = 6000):
        self.max_context_chars = max_context_chars

    def truncate_trace(self, trace_text: str) -> tuple[str, bool]:
        if len(trace_text) <= self.max_context_chars:
            return trace_text, False

        truncated = "[context truncated]\n" + trace_text[-self.max_context_chars :]
        return truncated, True

    def assemble(
        self,
        question: str,
        trace_text: str,
        memory_context: str = "",
    ) -> AssembledContext:
        trace_context, context_truncated = self.truncate_trace(trace_text)
        prompt = build_prompt(
            question=question,
            trace_text=trace_context,
            memory_context=memory_context,
        )

        return AssembledContext(
            prompt=prompt,
            trace_context=trace_context,
            memory_context=memory_context,
            context_chars=len(trace_context),
            context_truncated=context_truncated,
            memory_chars=len(memory_context),
            memory_used=bool(memory_context),
        )
