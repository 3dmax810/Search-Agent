import re
from dataclasses import dataclass


@dataclass
class AnswerConstraintResult:
    accepted: bool
    reason: str
    answer_token_count: int
    cleaned_answer: str


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\[\d+\]", "", answer)
    answer = re.sub(r"</?answer>", "", answer)
    return " ".join(answer.split()).strip(" .")


def count_answer_tokens(answer: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", clean_answer(answer)))


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _looks_like_organization(answer: str) -> bool:
    answer_norm = _normalize(answer)
    org_terms = {
        "center",
        "centre",
        "company",
        "corporation",
        "corp",
        "inc",
        "ltd",
        "limited",
        "university",
        "institute",
        "agency",
        "department",
        "pictures",
        "studios",
        "records",
        "network",
        "group",
        "committee",
        "association",
        "organization",
        "organisation",
    }
    return any(re.search(rf"\b{term}\b", answer_norm) for term in org_terms)


def _has_bridge_phrase(answer: str) -> bool:
    answer_lower = answer.lower()
    bridge_patterns = [
        r"\bthe author\b",
        r"\bwas written by\b",
        r"\bwritten by\b",
        r"\bthe performer\b",
        r"\bperformed by\b",
        r"\bthe company\b",
        r"\bthe distributor\b",
        r"\bdistributed by\b",
        r"\bthe manufacturer\b",
        r"\bthe owner\b",
        r"\bthe employer\b",
    ]
    return any(re.search(pattern, answer_lower) for pattern in bridge_patterns)


def _is_intermediate_entity_answer(question: str, cleaned_answer: str) -> bool:
    question_norm = _normalize(question)

    if _has_bridge_phrase(cleaned_answer):
        return True

    asks_for_location = any(
        phrase in question_norm
        for phrase in (
            "where",
            "which city",
            "what city",
            "what administrative territorial entity",
            "headquartered",
            "located",
            "birthplace",
        )
    )
    if asks_for_location and _looks_like_organization(cleaned_answer):
        return True

    asks_for_person = question_norm.startswith("who ") or " who " in question_norm
    asks_for_founder_or_spouse = any(
        phrase in question_norm
        for phrase in (
            "founded",
            "founder",
            "spouse",
            "wife",
            "husband",
            "grandmother",
            "grandfather",
        )
    )
    if asks_for_person and asks_for_founder_or_spouse and _looks_like_organization(
        cleaned_answer
    ):
        return True

    return False


def validate_answer_constraints(
    question: str,
    answer: str,
    max_answer_tokens: int | None = None,
    reject_intermediate_answers: bool = True,
) -> AnswerConstraintResult:
    cleaned = clean_answer(answer)
    token_count = count_answer_tokens(answer)

    if max_answer_tokens is not None and token_count > max_answer_tokens:
        return AnswerConstraintResult(
            accepted=False,
            reason=(
                f"answer too long: {token_count} tokens; "
                f"maximum allowed is {max_answer_tokens}"
            ),
            answer_token_count=token_count,
            cleaned_answer=cleaned,
        )

    if reject_intermediate_answers and _is_intermediate_entity_answer(
        question,
        cleaned,
    ):
        return AnswerConstraintResult(
            accepted=False,
            reason="answer appears to be an intermediate entity, not the requested final attribute",
            answer_token_count=token_count,
            cleaned_answer=cleaned,
        )

    return AnswerConstraintResult(
        accepted=True,
        reason="answer satisfies runtime answer constraints",
        answer_token_count=token_count,
        cleaned_answer=cleaned,
    )
