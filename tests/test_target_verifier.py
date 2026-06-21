import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.agent.target_verifier import (
    AnswerType,
    TargetType,
    infer_answer_type,
    infer_question_target_type,
    verify_target_answer,
)


def test_infers_question_target_type():
    assert (
        infer_question_target_type(
            "What is the seat of the county where Van Hook Township is located?"
        )
        == TargetType.COUNTY_SEAT
    )
    assert (
        infer_question_target_type(
            "What instrument is played by the person from The Blackout All-Stars?"
        )
        == TargetType.INSTRUMENT
    )


def test_infers_answer_type():
    assert infer_answer_type("Mountrail County, North Dakota. [1]") == AnswerType.COUNTY
    assert infer_answer_type("Ray Barretto. [1]") == AnswerType.PERSON
    assert infer_answer_type("conga. [1]") == AnswerType.INSTRUMENT
    assert infer_answer_type("Holley, New York. [1]") == AnswerType.SPECIFIC_LOCATION


def test_county_seat_question_rejects_county_answer():
    result = verify_target_answer(
        question="What is the seat of the county where Van Hook Township is located?",
        answer="Mountrail County, North Dakota. [1]",
    )

    assert result.accepted is False
    assert result.target_attribute == "county_seat"
    assert result.answer_type == "county"
    assert result.suggested_query == "Mountrail County, North Dakota county seat"


def test_instrument_question_rejects_person_answer():
    result = verify_target_answer(
        question="What instrument is played by the person from The Blackout All-Stars?",
        answer="Ray Barretto. [1]",
    )

    assert result.accepted is False
    assert result.target_attribute == "instrument"
    assert result.answer_type == "person"
    assert result.suggested_query == "Ray Barretto instrument"


def test_instrument_question_accepts_instrument_answer():
    result = verify_target_answer(
        question="What instrument is played by the person from The Blackout All-Stars?",
        answer="conga. [1]",
    )

    assert result.accepted is True


def test_county_question_rejects_city_state_answer():
    result = verify_target_answer(
        question="In what county is William W. Blair's birthplace located?",
        answer="Holley, New York. [1]",
    )

    assert result.accepted is False
    assert result.target_attribute == "county"
    assert result.answer_type == "specific_location"
    assert result.suggested_query == "Holley, New York county"


def test_record_label_question_accepts_label_answer():
    result = verify_target_answer(
        question="What record label is the performer of Almost Made Ya signed to?",
        answer="Universal Records. [1]",
    )

    assert result.accepted is True
