import re
from dataclasses import dataclass
from enum import Enum


class TargetType(str, Enum):
    UNKNOWN = "unknown"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    COUNTY = "county"
    BOROUGH = "borough"
    PROVINCE = "province"
    COUNTY_SEAT = "county_seat"
    HEADQUARTERS_LOCATION = "headquarters_location"
    INSTRUMENT = "instrument"
    RECORD_LABEL = "record_label"
    AWARD = "award"
    WORK = "work"


class AnswerType(str, Enum):
    UNKNOWN = "unknown"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    COUNTY = "county"
    BOROUGH = "borough"
    PROVINCE = "province"
    SPECIFIC_LOCATION = "specific_location"
    INSTRUMENT = "instrument"
    RECORD_LABEL = "record_label"
    AWARD = "award"
    WORK = "work"


@dataclass
class TargetVerificationResult:
    accepted: bool
    reason: str
    target_attribute: str
    answer_type: str
    suggested_query: str


def clean_answer(answer: str) -> str:
    answer = re.sub(r"\[\d+\]", "", answer)
    answer = re.sub(r"</?answer>", "", answer)
    return " ".join(answer.split()).strip(" .")


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def infer_question_target_type(question: str) -> TargetType:
    q = _normalize(question)
    patterns = [
        (r"\bcounty seat\b|\bseat of the county\b", TargetType.COUNTY_SEAT),
        (r"\bheadquartered\b|\bheadquarters\b", TargetType.HEADQUARTERS_LOCATION),
        (r"\bwhich county\b|\bwhat county\b|\bin what county\b", TargetType.COUNTY),
        (r"\bwhich borough\b|\bwhat borough\b|\bin which borough\b", TargetType.BOROUGH),
        (r"\bprovince shares a border\b|\bborder with\b", TargetType.PROVINCE),
        (r"\binstrument\b", TargetType.INSTRUMENT),
        (r"\brecord label\b", TargetType.RECORD_LABEL),
        (r"\bfounded\b|\bfounder\b", TargetType.PERSON),
        (r"\bspouse\b|\bwife\b|\bhusband\b", TargetType.PERSON),
        (r"\bgrandmother\b|\bgrandfather\b|\bfather\b|\bchild\b", TargetType.PERSON),
        (r"\bnotable work\b", TargetType.WORK),
        (r"\baward\b", TargetType.AWARD),
        (r"\bwhere\b|\blocated\b|\bbirthplace\b|\bborn in\b", TargetType.LOCATION),
        (r"\bcompany\b|\borganization\b|\borganisation\b|\binstitution\b", TargetType.ORGANIZATION),
        (r"^who\b|\bwho is\b|\bwho was\b", TargetType.PERSON),
    ]
    for pattern, target_type in patterns:
        if re.search(pattern, q):
            return target_type
    return TargetType.UNKNOWN


def _has_any_term(text: str, terms: set[str]) -> bool:
    text_norm = _normalize(text)
    return any(re.search(rf"\b{term}\b", text_norm) for term in terms)


def _looks_like_person(answer: str) -> bool:
    cleaned = clean_answer(answer)
    tokens = re.findall(r"[A-Z][a-zA-Z'.-]+", cleaned)
    if len(tokens) < 2:
        return False

    non_person_terms = {
        "Academy",
        "Award",
        "Borough",
        "City",
        "College",
        "Company",
        "Corporation",
        "County",
        "Entertainment",
        "High",
        "Province",
        "Records",
        "School",
        "Studios",
        "University",
    }
    return not any(token in non_person_terms for token in tokens)


def infer_answer_type(answer: str) -> AnswerType:
    cleaned = clean_answer(answer)

    if _has_any_term(cleaned, {"county"}):
        return AnswerType.COUNTY
    if _has_any_term(cleaned, {"borough"}):
        return AnswerType.BOROUGH
    if _has_any_term(cleaned, {"province"}):
        return AnswerType.PROVINCE
    if _has_any_term(
        cleaned,
        {
            "accordion",
            "bass",
            "cello",
            "clarinet",
            "conga",
            "congas",
            "drum",
            "drums",
            "flute",
            "guitar",
            "harmonica",
            "keyboard",
            "keyboards",
            "organ",
            "percussion",
            "piano",
            "saxophone",
            "sitar",
            "trombone",
            "trumpet",
            "violin",
            "vocals",
        },
    ):
        return AnswerType.INSTRUMENT
    if _has_any_term(cleaned, {"records", "record label", "label"}):
        return AnswerType.RECORD_LABEL
    if _has_any_term(cleaned, {"award", "oscar", "emmy", "grammy"}):
        return AnswerType.AWARD
    if _has_any_term(
        cleaned,
        {
            "company",
            "corporation",
            "corp",
            "inc",
            "pictures",
            "studios",
            "university",
            "college",
            "school",
            "center",
            "centre",
            "institute",
            "association",
            "committee",
            "entertainment",
        },
    ):
        return AnswerType.ORGANIZATION
    if "," in cleaned:
        return AnswerType.SPECIFIC_LOCATION
    if _looks_like_person(cleaned):
        return AnswerType.PERSON
    return AnswerType.UNKNOWN


def is_compatible(target_type: TargetType, answer_type: AnswerType) -> bool:
    if target_type == TargetType.UNKNOWN or answer_type == AnswerType.UNKNOWN:
        return True

    compatible_answer_types = {
        TargetType.PERSON: {AnswerType.PERSON},
        TargetType.ORGANIZATION: {
            AnswerType.ORGANIZATION,
            AnswerType.RECORD_LABEL,
        },
        TargetType.LOCATION: {
            AnswerType.LOCATION,
            AnswerType.SPECIFIC_LOCATION,
            AnswerType.COUNTY,
            AnswerType.BOROUGH,
            AnswerType.PROVINCE,
        },
        TargetType.COUNTY: {AnswerType.COUNTY},
        TargetType.BOROUGH: {AnswerType.BOROUGH},
        TargetType.PROVINCE: {AnswerType.PROVINCE},
        TargetType.COUNTY_SEAT: {
            AnswerType.LOCATION,
            AnswerType.SPECIFIC_LOCATION,
            AnswerType.UNKNOWN,
        },
        TargetType.HEADQUARTERS_LOCATION: {
            AnswerType.LOCATION,
            AnswerType.SPECIFIC_LOCATION,
            AnswerType.COUNTY,
            AnswerType.BOROUGH,
            AnswerType.PROVINCE,
            AnswerType.UNKNOWN,
        },
        TargetType.INSTRUMENT: {AnswerType.INSTRUMENT},
        TargetType.RECORD_LABEL: {AnswerType.RECORD_LABEL, AnswerType.ORGANIZATION},
        TargetType.AWARD: {AnswerType.AWARD, AnswerType.UNKNOWN},
        TargetType.WORK: {AnswerType.WORK, AnswerType.UNKNOWN},
    }
    return answer_type in compatible_answer_types.get(target_type, {AnswerType.UNKNOWN})


def build_next_query(answer: str, target_type: TargetType) -> str:
    cleaned = clean_answer(answer)
    cleaned = re.sub(r"\bplays in\b.*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bwas born in\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bis located in\b", "", cleaned, flags=re.I)
    cleaned = " ".join(cleaned.split()).strip(" .")
    target_query = target_type.value.replace("_", " ")
    if not cleaned:
        return target_query
    return f"{cleaned} {target_query}"


def verify_target_answer(question: str, answer: str) -> TargetVerificationResult:
    target_type = infer_question_target_type(question)
    if target_type == TargetType.UNKNOWN:
        return TargetVerificationResult(
            accepted=True,
            reason="question target type is unknown; skip target verification",
            target_attribute=target_type.value,
            answer_type=AnswerType.UNKNOWN.value,
            suggested_query="",
        )

    answer_type = infer_answer_type(answer)
    if is_compatible(target_type, answer_type):
        return TargetVerificationResult(
            accepted=True,
            reason="answer type is compatible with question target type",
            target_attribute=target_type.value,
            answer_type=answer_type.value,
            suggested_query="",
        )

    return TargetVerificationResult(
        accepted=False,
        reason=(
            "answer type is incompatible with question target type: "
            f"target={target_type.value}, answer={answer_type.value}"
        ),
        target_attribute=target_type.value,
        answer_type=answer_type.value,
        suggested_query=build_next_query(answer, target_type),
    )
