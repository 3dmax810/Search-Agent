import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

import re

from search_agent.agent.parser import parse_model_output

# from search_agent.agent.prompts import build_prompt
from search_agent.agent.context import ContextAssembler
from search_agent.agent.constraints import validate_answer_constraints
from search_agent.agent.target_verifier import verify_target_answer
from search_agent.agent.llm import OllamaModel
from search_agent.tools.search_tool import LocalSearchTool
from search_agent.agent.judge import AnswerJudge
from search_agent.memory.store import MemoryStore


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _normalize_query(query: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return " ".join(tokens)


def _query_signature(query: str) -> str:
    stopwords = {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "to",
        "for",
        "with",
        "in",
        "on",
        "by",
    }
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if token not in stopwords
    ]
    return " ".join(sorted(tokens))


def _build_search_control_context(
    searched_queries: list[str],
    duplicate_search_count: int,
) -> str:
    if not searched_queries:
        return ""

    query_lines = "\n".join(f"- {query}" for query in searched_queries)
    duplicate_note = ""
    if duplicate_search_count:
        duplicate_note = (
            f"\nDuplicate search attempts so far: {duplicate_search_count}.\n"
        )

    return f"""

Search control:
Already searched queries. Do not repeat these queries or reorder the same words:
{query_lines}
{duplicate_note}
Next search must introduce a new named entity from observations, a new missing attribute, or a clearly different keyword set. If the current evidence is sufficient, answer instead of searching again.
"""


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
        memory_store=None,
        context_assembler=None,
        max_search_turns: int = 3,
        max_model_turns: int = 8,
        max_context_chars: int = 6000,
        max_duplicate_searches: int = 2,
        max_answer_tokens: int | None = None,
        reject_intermediate_answers: bool = True,
        use_target_verifier: bool = True,
        use_answer_judge: bool = True,
        use_memory: bool = True,
    ):
        self.model = model or OllamaModel()
        self.search_tool = search_tool or LocalSearchTool()
        self.answer_judge = answer_judge or AnswerJudge()
        self.memory_store = memory_store or MemoryStore()
        self.context_assembler = context_assembler or ContextAssembler(
            max_context_chars=max_context_chars
        )
        self.use_answer_judge = use_answer_judge
        self.use_memory = use_memory
        self.max_search_turns = max_search_turns
        self.max_model_turns = max_model_turns
        self.max_context_chars = max_context_chars
        self.max_duplicate_searches = max_duplicate_searches
        self.max_answer_tokens = max_answer_tokens
        self.reject_intermediate_answers = reject_intermediate_answers
        self.use_target_verifier = use_target_verifier

    def run(self, question: str) -> dict:
        empty_output_count = 0
        max_empty_outputs = 2

        trace = []
        trace_text = ""
        seen_search_queries = set()
        seen_search_signatures = set()
        searched_queries = []
        duplicate_search_count = 0

        memory_context = ""
        if self.use_memory:
            memory_context = self.memory_store.build_memory_context(question)

        search_turns = 0
        for turn in range(self.max_model_turns + 1):
            # context = _truncate_context(trace_text, self.max_context_chars)
            # prompt = build_prompt(question, context)
            runtime_trace_text = (
                trace_text
                + _build_search_control_context(
                    searched_queries=searched_queries,
                    duplicate_search_count=duplicate_search_count,
                )
            )
            assembled_context = self.context_assembler.assemble(
                question=question,
                trace_text=runtime_trace_text,
                memory_context=memory_context,
            )
            prompt = assembled_context.prompt
            model_output = self.model.generate(prompt)
            if not model_output.strip():
                empty_output_count += 1
                step = {
                    "turn": turn,
                    "search_turns": search_turns,
                    "max_search_turns": self.max_search_turns,
                    "context_chars": assembled_context.context_chars,
                    "context_truncated": assembled_context.context_truncated,
                    "prompt_chars": len(prompt),
                    "prompt": prompt,
                    "model_output": model_output,
                    "action": "error",
                    "content": "",
                    "error": "empty model output",
                    "state": "PARSED",
                    "halt_reason": "",
                    "empty_output_count": empty_output_count,
                    "fallback": "retry without adding empty output to trace context",
                    "memory_chars": assembled_context.memory_chars,
                    "memory_used": assembled_context.memory_used,
                    "searched_queries": searched_queries.copy(),
                    "duplicate_search_count": duplicate_search_count,
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
                "context_chars": assembled_context.context_chars,
                "context_truncated": assembled_context.context_truncated,
                "prompt_chars": len(prompt),
                "prompt": prompt,
                "model_output": model_output,
                "action": parsed.action,
                "content": parsed.content,
                "error": parsed.error,
                "state": "PARSED",
                "halt_reason": "",
                "memory_chars": assembled_context.memory_chars,
                "memory_used": assembled_context.memory_used,
                "searched_queries": searched_queries.copy(),
                "duplicate_search_count": duplicate_search_count,
            }

            if parsed.action == "answer":
                step["state"] = "ANSWER_CHECK"
                constraint_result = validate_answer_constraints(
                    question=question,
                    answer=parsed.content,
                    max_answer_tokens=self.max_answer_tokens,
                    reject_intermediate_answers=self.reject_intermediate_answers,
                )
                step["answer_token_count"] = constraint_result.answer_token_count
                step["answer_constraint_accepted"] = constraint_result.accepted
                step["answer_constraint_reason"] = constraint_result.reason
                if not constraint_result.accepted:
                    step["accepted"] = False
                    step["reject_reason"] = constraint_result.reason
                    step["state"] = "ANSWER_REJECTED"
                    trace.append(step)
                    trace_text += (
                        f"\nAnswer rejected: {constraint_result.reason}. "
                        "If the evidence already contains the final answer, output only "
                        "the shortest final answer span with citation. If the answer is "
                        "only an intermediate entity, search for the missing final attribute.\n"
                    )
                    continue

                if self.use_target_verifier:
                    target_result = verify_target_answer(
                        question=question,
                        answer=parsed.content,
                    )
                    step["target_verifier_accepted"] = target_result.accepted
                    step["target_verifier_reason"] = target_result.reason
                    step["target_attribute"] = target_result.target_attribute
                    step["answer_type"] = target_result.answer_type
                    step["target_verifier_suggested_query"] = (
                        target_result.suggested_query
                    )
                    if not target_result.accepted:
                        step["accepted"] = False
                        step["reject_reason"] = target_result.reason
                        step["state"] = "ANSWER_REJECTED"
                        trace.append(step)
                        next_action = ""
                        if target_result.suggested_query:
                            next_action = (
                                f" Next action should be "
                                f"<search>{target_result.suggested_query}</search>."
                            )
                        trace_text += (
                            f"\nAnswer rejected: {target_result.reason}.{next_action} "
                            "Search for the requested final attribute instead of "
                            "answering with the intermediate entity.\n"
                        )
                        continue

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
                    step["state"] = "ANSWER_ACCEPTED"
                    trace.append(step)
                    result = {
                        "answer": parsed.content,
                        "trace": trace,
                    }

                    if self.use_memory:
                        self.memory_store.save_from_agent_result(
                            question=question,
                            result=result,
                        )
                    return result

                step["state"] = "ANSWER_REJECTED"
                trace.append(step)
                trace_text += (
                    f"\nAnswer rejected: {reject_reason}. "
                    "If the current evidence does not contain the requested final answer, "
                    "you must issue another <search> for the missing entity or attribute. "
                    "Do not answer that the evidence is insufficient while search turns remain.\n"
                )
                continue

            if parsed.action == "search":
                step["state"] = "SEARCH"
                normalized_query = _normalize_query(parsed.content)
                query_signature = _query_signature(parsed.content)
                duplicate_reason = ""
                if normalized_query in seen_search_queries:
                    duplicate_reason = "same normalized query"
                elif query_signature and query_signature in seen_search_signatures:
                    duplicate_reason = "same query keyword set"

                if duplicate_reason:
                    duplicate_search_count += 1
                    step["state"] = "DUPLICATE_SEARCH"
                    step["error"] = "duplicate search query"
                    step["duplicate_reason"] = duplicate_reason
                    step["duplicate_search_count"] = duplicate_search_count
                    step["fallback"] = "ask model to reformulate the search query"
                    trace.append(step)
                    searched = "\n".join(f"- {q}" for q in searched_queries)
                    trace_text += (
                        f"\nSearch rejected: duplicate query '{parsed.content}'. "
                        f"Reason: {duplicate_reason}.\n"
                        f"Already searched queries:\n{searched}\n"
                        "Do not repeat or reorder the same query. The next search must use "
                        "a new named entity from the observations, a new missing attribute, "
                        "or a clearly different keyword set.\n"
                    )
                    if duplicate_search_count >= self.max_duplicate_searches:
                        trace_text += (
                            "Repeated duplicate searches have reached the duplicate limit. "
                            "If no genuinely new query is available, stop searching and "
                            "provide the best supported <answer> from existing observations.\n"
                        )
                    continue

                if search_turns >= self.max_search_turns:
                    step["state"] = "HALT"
                    step["halt_reason"] = "search limit reached"
                    step["fallback"] = "search limit reached"
                    trace.append(step)
                    trace_text += (
                        "\nSearch limit reached. You must provide a supported <answer> "
                        "based on existing observations.\n"
                    )
                    continue

                search_turns += 1
                seen_search_queries.add(normalized_query)
                if query_signature:
                    seen_search_signatures.add(query_signature)
                searched_queries.append(parsed.content)
                step["search_turns"] = search_turns
                step["searched_queries"] = searched_queries.copy()
                results = self.search_tool.search(parsed.content)
                step["search_results"] = [r.__dict__ for r in results]
                trace.append(step)

                observe = "\n".join(
                    f"[{i + 1}] {r.title}: {r.snippet} ({r.url})"
                    for i, r in enumerate(results)
                )
                trace_text += f"\n{model_output}\n<observe>{observe}</observe>\n"
                continue

            step["state"] = "FORMAT_ERROR"
            step["fallback"] = "format error, ask model to retry"
            trace.append(step)
            trace_text += (
                f"\nFormat error: {parsed.error}. Please retry with valid tags.\n"
            )
        return {
            "answer": "I could not produce a valid answer within the model turn limit.",
            "trace": trace,
        }
