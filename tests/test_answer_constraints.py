import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.agent.constraints import validate_answer_constraints


def test_short_final_answer_passes():
    result = validate_answer_constraints(
        question="Which city is the birthplace of the author of The Silent Harbor?",
        answer="Brookhaven. [1]",
        max_answer_tokens=8,
    )

    assert result.accepted is True
    assert result.answer_token_count == 1


def test_long_explanatory_answer_is_rejected():
    result = validate_answer_constraints(
        question="Which city is the birthplace of the author of The Silent Harbor?",
        answer="The birthplace of the author of The Silent Harbor is Brookhaven. [1]",
        max_answer_tokens=8,
    )

    assert result.accepted is False
    assert "answer too long" in result.reason


def test_founder_question_rejects_company_as_intermediate_entity():
    result = validate_answer_constraints(
        question="Who founded the company that distributed the film UHF?",
        answer="Orion Pictures. [1]",
        max_answer_tokens=8,
    )

    assert result.accepted is False
    assert "intermediate entity" in result.reason


def test_location_question_rejects_organization_as_intermediate_entity():
    result = validate_answer_constraints(
        question="Where is Ulrich Walter's employer headquartered?",
        answer="German Aerospace Center. [1]",
        max_answer_tokens=8,
    )

    assert result.accepted is False
    assert "intermediate entity" in result.reason
