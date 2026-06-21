import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryItem:
    question: str
    answer: str
    search_queries: list[str]
    trace_summary: str
    success: bool = True


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _extract_search_queries(trace: list[dict]) -> list[str]:
    queries = []
    for step in trace:
        if step.get("action") == "search":
            content = step.get("content", "").strip()
            if content:
                queries.append(content)
    return queries


def _build_trace_summary(search_queries: list[str], answer: str) -> str:
    if not search_queries:
        return f"The agent answered without issuing a search. Final answer: {answer}"
    query_text = " -> ".join(search_queries)
    return f"The agent searched: {query_text}. Final answer: {answer}"


class MemoryStore:
    def __init__(self, memory_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]

        if memory_path is None:
            memory_path = project_root / "data" / "memory" / "agent_memory.jsonl"
        self.memory_path = Path(memory_path)

    def load_all(self) -> list[MemoryItem]:
        if not self.memory_path.exists():
            return []

        items = []
        with self.memory_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                items.append(
                    MemoryItem(
                        question=data["question"],
                        answer=data["answer"],
                        search_queries=data.get("search_queries", []),
                        trace_summary=data.get("trace_summary", ""),
                        success=data.get("success", True),
                    )
                )
        return items

    def save(self, item: MemoryItem) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "question": item.question,
            "answer": item.answer,
            "search_queries": item.search_queries,
            "trace_summary": item.trace_summary,
            "success": item.success,
        }

        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_from_agent_result(self, question: str, result: dict) -> None:
        answer = result.get("answer", "")
        trace = result.get("trace", [])
        if not answer or not trace:
            return

        last_step = trace[-1]
        if last_step.get("action") != "answer":
            return

        if last_step.get("accepted") is False:
            return

        search_queries = _extract_search_queries(trace)
        trace_summary = _build_trace_summary(search_queries, answer)
        item = MemoryItem(
            question=question,
            answer=answer,
            search_queries=search_queries,
            trace_summary=trace_summary,
            success=True,
        )
        self.save(item)

    def search(self, question: str, top_k: int = 3) -> list[MemoryItem]:
        items = self.load_all()
        if not items:
            return []

        query_tokens = set(_tokenize(question))
        scored_items = []

        for item in items:
            memory_tokens = set(_tokenize(item.question))
            overlap = len(query_tokens & memory_tokens)

            if overlap > 0:
                scored_items.append((overlap, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:top_k]]

    def build_memory_context(self, question: str, top_k: int = 3) -> str:
        memories = self.search(question, top_k=top_k)
        if not memories:
            return ""

        lines = [
            "Relevant memory:",
            "These are past search strategies. Do not use them as evidence.",
            "Final answers must still be supported by current <observe> content.",
        ]

        for i, memory in enumerate(memories, start=1):
            lines.append(f"[Memory {i}]")
            lines.append(f"Past question: {memory.question}")
            lines.append(f"Search strategy: {' -> '.join(memory.search_queries)}")
            lines.append(f"Trace summary: {memory.trace_summary}")

        return "\n".join(lines)
