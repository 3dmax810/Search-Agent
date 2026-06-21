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
    search_action_turns: int = 0
    model_turns: int = 0
    answer_reject_count: int = 0
    duplicate_search_count: int = 0
    search_limit_count: int = 0
    format_error_count: int = 0
    answer_constraint_reject_count: int = 0
    target_verifier_reject_count: int = 0
    final_state: str = ""


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
            search_action_turns=0,
            model_turns=1,
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
            search_action_turns=1,
            model_turns=1,
        )


class SearchAgentBaseline:
    def __init__(self, agent: AgentRuntime | None = None):
        self.agent = agent or AgentRuntime()

    def run(self, question: str) -> BaselineResult:
        result = self.agent.run(question)

        answer = result.get("answer", "")
        trace = result.get("trace", [])

        real_search_turns = sum(
            1
            for step in trace
            if step.get("action") == "search"
            and step.get("state") == "SEARCH"
            and step.get("search_results")
        )
        search_action_turns = sum(
            1 for step in trace if step.get("action") == "search"
        )
        answer_reject_count = sum(
            1 for step in trace if step.get("state") == "ANSWER_REJECTED"
        )
        duplicate_search_count = sum(
            1 for step in trace if step.get("state") == "DUPLICATE_SEARCH"
        )
        search_limit_count = sum(
            1
            for step in trace
            if step.get("state") == "HALT"
            and step.get("halt_reason") == "search limit reached"
        )
        format_error_count = sum(
            1 for step in trace if step.get("state") == "FORMAT_ERROR"
        )
        answer_constraint_reject_count = sum(
            1
            for step in trace
            if step.get("state") == "ANSWER_REJECTED"
            and step.get("answer_constraint_accepted") is False
        )
        target_verifier_reject_count = sum(
            1
            for step in trace
            if step.get("state") == "ANSWER_REJECTED"
            and step.get("target_verifier_accepted") is False
        )
        final_state = trace[-1].get("state", "") if trace else ""

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
            search_turns=real_search_turns,
            retrieved_docs=retrieved_docs,
            search_action_turns=search_action_turns,
            model_turns=len(trace),
            answer_reject_count=answer_reject_count,
            duplicate_search_count=duplicate_search_count,
            search_limit_count=search_limit_count,
            format_error_count=format_error_count,
            answer_constraint_reject_count=answer_constraint_reject_count,
            target_verifier_reject_count=target_verifier_reject_count,
            final_state=final_state,
        )
