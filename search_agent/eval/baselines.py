import json
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

import re
from dataclasses import dataclass
from typing import Protocol

from search_agent.agent.llm import OllamaModel
from search_agent.tools.search_tool import LocalSearchTool, SearchResults
from search_agent.agent.runtime import AgentRuntime


class ModelClient(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class BaselineResult:
    method: str
    question: str
    answer: str
    raw_output: str
    citations: list[str]
    search_turns: int
    retrieved_docs: list[dict]


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if match:
        return match.group(1).strip()

    return text.strip()


def _extract_citations(answer: str) -> list[str]:
    return re.findall(r"\[(\d+)\]", answer)


def _format_search_results(results: list[SearchResults]) -> str:
    return "\n".join(
        f"[{i + 1}]{result.title}: {result.snippet} ({result.url})"
        for i, result in enumerate(results)
    )


class NoSearchBaseline:
    def __init__(self, model: ModelClient | None = None):
        self.model = model or OllamaModel()

    def run(self, question: str) -> BaselineResult:
        prompt = f"""Answer the question directly.

        Rules:
        - Output only one <answer>...</answer> tag.
        - Do not use search.
        - If you are unsure, still provide your best short answer.
        
        Question:
        {question}

        Response:
        """

        raw_output = self.model.generate(prompt)
        answer = _extract_answer(raw_output)

        return BaselineResult(
            method="no-search",
            question=question,
            answer=answer,
            raw_output=raw_output,
            citations=_extract_citations(answer),
            search_turns=0,
            retrieved_docs=[],
        )


class SingleShotRAGBaseline:
    def __init__(
        self,
        model: ModelClient | None = None,
        search_tool: LocalSearchTool | None = None,
        top_k: int = 3,
    ):
        self.model = model or OllamaModel()
        self.search_tool = search_tool or LocalSearchTool()
        self.top_k = top_k

    def run(self, question: str) -> BaselineResult:
        results = self.search_tool.search(question, top_k=self.top_k)
        evidence = _format_search_results(results)

        prompt = f"""Answer the question using only the evidence below.
        Rules:
        - Output only one <answer>...</answer> tag.
        - The answer must include citation numbers like [1].
        - Do not use facts that are not supported by the evidence.

        Question:
        {question}

        Evidence:
        {evidence}

        Response:
        """

        raw_output = self.model.generate(prompt)
        answer = _extract_answer(raw_output)
        return BaselineResult(
            method="single-shot-rag",
            question=question,
            answer=answer,
            raw_output=raw_output,
            citations=_extract_citations(answer),
            search_turns=1,
            retrieved_docs=[result.__dict__ for result in results],
        )


class SearchAgentBaseline:
    def __init__(self, agent: AgentRuntime | None = None):
        self.agent = agent or AgentRuntime()

    def run(self, question: str) -> BaselineResult:
        result = self.agent.run(question)

        answer = result.get("answer", "")
        trace = result.get("trace", [])

        search_turns = sum(1 for step in trace if step.get("action") == "search")

        retrieved_docs = []
        for step in reversed(trace):
            if step.get("action") == "search" and step.get("search_results"):
                retrieved_docs = step["search_results"]
                break
        # for step in trace:
        #     for doc in step.get("search_results", []):
        #         retrieved_docs.append(doc)

        return BaselineResult(
            method="search-agent",
            question=question,
            answer=answer,
            raw_output=json.dumps(result, ensure_ascii=False),
            citations=_extract_citations(answer),
            search_turns=search_turns,
            retrieved_docs=retrieved_docs,
        )
