import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

import re

from search_agent.agent.parser import parse_model_output
from search_agent.agent.prompts import build_prompt
from search_agent.agent.llm import OllamaModel
from search_agent.tools.search_tool import LocalSearchTool
from search_agent.agent.judge import AnswerJudge


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_observe_text(trace_text: str) -> str:
    observes = re.findall(r"<observe>(.*?)</observe>", trace_text, flags=re.S)
    return "\n".join(observes)


def _answer_supported_by_observe(answer: str, trace_text: str) -> tuple[bool, str]:
    observe_text = _normalize_text(_extract_observe_text(trace_text))
    answer_text = _normalize_text(answer)
    insufficient_phrases = [
        "does not state",
        "doesn't state",
        "not state",
        "not provided",
        "not mentioned",
        "cannot determine",
        "can't determine",
        "insufficient",
        "unknown",
        "not enough information",
    ]

    if any(phrase in answer_text for phrase in insufficient_phrases):
        return False, "answer says evidence is insufficient; continue searching"
    if not observe_text:
        return False, "no observe evidence available"
    citation_matches = re.findall(r"\[(\d+)\]", answer)
    if not citation_matches:
        return False, "answer missing citation"

    cleaned = re.sub(r"\[\d+\]", "", answer_text)
    tokens = re.findall(r"[a-z0-9]+", cleaned)

    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "who",
        "which",
        "what",
        "is",
        "was",
        "were",
        "in",
        "on",
        "by",
        "to",
        "for",
        "with",
        "author",
        "born",
        "birthplace",
        "city",
    }

    keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
    if not keywords:
        return False, "no meaningful answer keywords found"

    missing = [kw for kw in keywords if kw not in observe_text]
    hit_count = len(keywords) - len(missing)
    if hit_count / len(keywords) < 0.5:
        return (
            False,
            f"answer not supported by observe, missing keywords: {missing[:5]}",
        )

    return True, ""


def _truncate_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return "[context truncated]\n" + text[-max_chars:]


class FakeModel:
    def generate(self, prompt: str) -> str:
        if "The Old Man and the Sea is a novel by Ernest Hemingway" not in prompt:
            return "<think>I need to find the author.</think><search>author of The Old Man and the Sea</search>"

        if "Ernest Hemingway was born in Oak Park, Illinois" not in prompt:
            return "<think>I know the author is Hemingway. Now find his birthplace.</think><search>Ernest Hemingway birthplace</search>"

        return "<answer>Ernest Hemingway was born in Oak Park, Illinois. [1]</answer>"


class AgentRuntime:
    def __init__(
        self,
        model=None,
        search_tool=None,
        answer_judge=None,
        max_search_turns: int = 3,
        max_model_turns: int = 8,
        max_context_chars: int = 6000,
        use_answer_judge: bool = True,
    ):
        self.model = model or OllamaModel()
        self.search_tool = search_tool or LocalSearchTool()
        self.answer_judge = answer_judge or AnswerJudge()
        self.use_answer_judge = use_answer_judge
        self.max_search_turns = max_search_turns
        self.max_model_turns = max_model_turns
        self.max_context_chars = max_context_chars

    def run(self, question: str) -> dict:
        empty_output_count = 0
        max_empty_outputs = 2

        trace = []
        trace_text = ""

        search_turns = 0
        for turn in range(self.max_model_turns + 1):
            context = _truncate_context(trace_text, self.max_context_chars)
            prompt = build_prompt(question, context)
            model_output = self.model.generate(prompt)
            if not model_output.strip():
                empty_output_count += 1
                step = {
                    "turn": turn,
                    "search_turns": search_turns,
                    "max_search_turns": self.max_search_turns,
                    "context_chars": len(context),
                    "context_truncated": len(trace_text) > self.max_context_chars,
                    "prompt_chars": len(prompt),
                    "prompt": prompt,
                    "model_output": model_output,
                    "action": "error",
                    "content": "",
                    "error": "empty model output",
                    "empty_output_count": empty_output_count,
                    "fallback": "retry without adding empty output to trace context",
                }
                trace.append(step)
                if empty_output_count >= max_empty_outputs:
                    trace_text += (
                        "\nThe previous model responses were empty."
                        "You must now output exactly one valid <search> or <answer> action.\n"
                    )
                continue

            parsed = parse_model_output(model_output)
            empty_output_count = 0
            step = {
                "turn": turn,
                "search_turns": search_turns,
                "max_search_turns": self.max_search_turns,
                "context_chars": len(context),
                "context_truncated": len(trace_text) > self.max_context_chars,
                "prompt_chars": len(prompt),
                "prompt": prompt,
                "model_output": model_output,
                "action": parsed.action,
                "content": parsed.content,
                "error": parsed.error,
            }

            if parsed.action == "answer":
                supported, reject_reason = _answer_supported_by_observe(
                    parsed.content,
                    trace_text,
                )
                step["accepted"] = supported
                step["reject_reason"] = reject_reason
                if supported and self.use_answer_judge:
                    observe_text = _extract_observe_text(trace_text)
                    judge_accepted, judge_reason = self.answer_judge.judge(
                        question=question,
                        answer=parsed.content,
                        observe_text=observe_text,
                    )

                    step["judge_accepted"] = judge_accepted
                    step["judge_reason"] = judge_reason
                    if not judge_accepted:
                        supported = False
                        reject_reason = f"answer judge rejected: {judge_reason}"
                        step["accepted"] = False
                        step["reject_reason"] = reject_reason

                if supported:
                    trace.append(step)
                    return {
                        "answer": parsed.content,
                        "trace": trace,
                    }

                trace.append(step)
                trace_text += (
                    f"\nAnswer rejected: {reject_reason}. "
                    "If the current evidence does not contain the requested final answer, "
                    "you must issue another <search> for the missing entity or attribute. "
                    "Do not answer that the evidence is insufficient while search turns remain.\n"
                )
                continue

            if parsed.action == "search":
                if search_turns >= self.max_search_turns:
                    step["fallback"] = "search limit reached"
                    trace.append(step)
                    trace_text += (
                        "\nSearch limit reached. You must provide a supported <answer> "
                        "based on existing observations.\n"
                    )
                    continue

                search_turns += 1
                step["search_turns"] = search_turns
                results = self.search_tool.search(parsed.content)
                step["search_results"] = [r.__dict__ for r in results]
                trace.append(step)

                observe = "\n".join(
                    f"[{i + 1}] {r.title}: {r.snippet} ({r.url})"
                    for i, r in enumerate(results)
                )
                trace_text += f"\n{model_output}\n<observe>{observe}</observe>\n"
                continue

            step["fallback"] = "format error, ask model to retry"
            trace.append(step)
            trace_text += (
                f"\nFormat error: {parsed.error}. Please retry with valid tags.\n"
            )
        return {
            "answer": "I could not produce a valid answer within the model turn limit.",
            "trace": trace,
        }
