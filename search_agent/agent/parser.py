#  从模型输出判断下一步是搜索还是回答
from re import search
import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class ParsedAction:
    action: Literal["search", "answer", "error"]  # 枚举类型
    content: str
    raw: str
    error: str | None = None


def _remove_allowed_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<search>.*?</search>", "", text, flags=re.S)
    text = re.sub(r"<answer>.*?</answer>", "", text, flags=re.S)
    return text.strip()


def parse_model_output(text: str) -> ParsedAction:
    search_matches = re.findall(r"<search>(.*?)</search>", text, re.S)
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if search_matches and answer_matches:
        return ParsedAction(
            "error", "", text, "model output contains both search and answer"
        )
    if len(search_matches) > 1:
        return ParsedAction(
            "error", "", text, "model output contains multiple search tags"
        )
    if len(answer_matches) > 1:
        return ParsedAction(
            "error", "", text, "model output contains multiple answer tags"
        )
    extra_text = _remove_allowed_tags(text)
    if extra_text:
        return ParsedAction(
            "error",
            "",
            text,
            f"model output contains text outside allowed tags: {extra_text[:80]}",
        )
    if search_matches:
        query = search_matches[0].strip()
        if not query:
            return ParsedAction("error", "", text, "empty search query")
        return ParsedAction("search", query, text)

    if answer_matches:
        final = answer_matches[0].strip()
        if not final:
            return ParsedAction("error", "", text, "empty answer")
        if not re.search(r"\[\d+\]", final):
            return ParsedAction("error", "", text, "answer missing citation")
        return ParsedAction("answer", final, text)

    return ParsedAction("error", "", text, "missing <search> or <answer>")
