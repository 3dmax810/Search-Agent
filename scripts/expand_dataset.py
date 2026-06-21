import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "data" / "processed" / "docs.jsonl"
QA_PATH = ROOT / "data" / "eval" / "qa.jsonl"
PROMPT_PATH = ROOT / "data" / "dataset_generation_prompt.txt"


FIRST_NAMES = [
    "Alden",
    "Bryn",
    "Celia",
    "Darin",
    "Elian",
    "Fara",
    "Galen",
    "Hessa",
    "Iven",
    "Jora",
    "Kael",
    "Liora",
    "Maren",
    "Nolan",
    "Orin",
    "Pella",
    "Quin",
    "Rhea",
    "Soren",
    "Talia",
    "Ulric",
    "Vera",
    "Wylan",
    "Xara",
    "Yorin",
    "Zella",
]

SURNAMES = [
    "Arven",
    "Bex",
    "Cairn",
    "Dale",
    "Ely",
    "Fenn",
    "Grove",
    "Hale",
    "Iris",
    "Joss",
    "Kade",
    "Lark",
    "Moss",
    "Nire",
    "Orr",
    "Pike",
    "Quell",
    "Rook",
    "Sable",
    "Thorn",
    "Umber",
    "Voss",
    "Wren",
    "Yale",
    "Zorin",
]

CITY_NAMES = [
    "Ashford",
    "Brindle",
    "Corvale",
    "Dunmere",
    "Eastridge",
    "Fenwick",
    "Greyhaven",
    "Highmere",
    "Ironbay",
    "Juniper",
    "Kingswell",
    "Lowmarsh",
    "Moonford",
    "Norvale",
    "Oakmere",
    "Pineholt",
    "Quarryden",
    "Rivermark",
    "Stonefield",
    "Thornwick",
    "Umberfall",
    "Valeport",
    "Westhaven",
    "Yarrowby",
    "Zephyrton",
]

INSTITUTIONS = [
    "Aster Hall Institute",
    "Briarfield University",
    "Cobalt Research College",
    "Dunwick Technical School",
    "Elderstone Laboratory",
    "Fenmark Academy",
    "Greywater Institute",
    "Harborline College",
    "Irongate Polytechnic",
    "Juniper Science House",
    "Keystone University",
    "Larkhaven Institute",
]

ORGANIZATIONS = [
    "Auralis Observatory",
    "Bexford Arts Bureau",
    "Cairnfield Research House",
    "Dovetail Archive",
    "Eonbridge Laboratory",
    "Farside Civic Museum",
    "Glimmerstone Conservatory",
    "Holloway Design Office",
    "Ivoryline Foundation",
    "Jadeport Survey Center",
]

ADJECTIVES = [
    "Amber",
    "Blue",
    "Copper",
    "Distant",
    "Emerald",
    "Frost",
    "Golden",
    "Hidden",
    "Ivory",
    "Jade",
    "Kindled",
    "Lunar",
    "Misty",
    "Northern",
    "Opal",
    "Pale",
    "Quiet",
    "Red",
    "Silver",
    "Twilight",
]

NOUNS = [
    "Harbor",
    "Compass",
    "Lantern",
    "Garden",
    "Engine",
    "Bridge",
    "Map",
    "Signal",
    "Mirror",
    "Orbit",
    "Meadow",
    "Archive",
    "Spire",
    "River",
    "Thread",
    "Kite",
    "Foundry",
    "Reef",
    "Crown",
    "Circuit",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def doc_num(doc_id: str) -> int:
    return int(doc_id.replace("doc", ""))


def person(i: int) -> str:
    return f"{FIRST_NAMES[i % len(FIRST_NAMES)]} {SURNAMES[(i * 7) % len(SURNAMES)]}"


def city(i: int) -> str:
    return CITY_NAMES[(i * 7) % len(CITY_NAMES)]


def institution(i: int) -> str:
    return INSTITUTIONS[(i * 5) % len(INSTITUTIONS)]


def organization(i: int) -> str:
    return ORGANIZATIONS[(i * 3) % len(ORGANIZATIONS)]


def title(i: int, suffix: str = "") -> str:
    base = f"{ADJECTIVES[i % len(ADJECTIVES)]} {NOUNS[(i * 3) % len(NOUNS)]}"
    return f"{base} {i:02d} {suffix}".strip()


class DatasetBuilder:
    def __init__(self, start_doc_id: int):
        self.next_doc_id = start_doc_id
        self.docs: list[dict] = []
        self.qa_rows: list[dict] = []
        self.generated_cases: list[dict] = []

    def add_doc(self, title_text: str, body: str) -> str:
        doc_id = f"doc{self.next_doc_id:03d}"
        self.next_doc_id += 1
        self.docs.append(
            {
                "doc_id": doc_id,
                "title": title_text,
                "text": body,
                "source": "synthetic",
            }
        )
        return doc_id

    def add_qa(self, question: str, answer: str, supporting_doc_ids: list[str]) -> None:
        self.qa_rows.append(
            {
                "question": question,
                "answer": answer,
                "supporting_doc_ids": supporting_doc_ids,
                "type": "multi-hop",
            }
        )

    def add_two_hop_case(self, case_index: int, pattern: str) -> None:
        p = person(case_index)
        answer_city = city(case_index)
        org = organization(case_index)
        inst = institution(case_index)

        if pattern == "book_birthplace":
            work = title(case_index, "Novel")
            d1 = self.add_doc(
                work,
                f"{work} is a novel written by {p}. The catalog describes a fictional expedition through numbered weather gates.",
            )
            d2 = self.add_doc(
                p,
                f"{p} was born in {answer_city}. Later interviews mention that city as the place where the author first kept travel journals.",
            )
            q = f"Which city is the birthplace of the author of {work}?"
            a = answer_city
            meta = {"pattern": pattern, "anchor": work, "answer": a, "docs": [d1, d2]}

        elif pattern == "film_actor_birthplace":
            work = title(case_index, "Film")
            d1 = self.add_doc(
                work,
                f"{work} is a film starring {p}. The film follows a courier who maps invented train stations across a glass desert.",
            )
            d2 = self.add_doc(
                p,
                f"{p} was born in {answer_city}. The performer's early stage notes list that city beside several amateur theater credits.",
            )
            q = f"Which city is the birthplace of the actor who starred in {work}?"
            a = answer_city
            meta = {"pattern": pattern, "anchor": work, "answer": a, "docs": [d1, d2]}

        elif pattern == "company_founder_birthplace":
            company = title(case_index, "Systems").replace(" ", "")
            d1 = self.add_doc(
                company,
                f"{company} was founded by {p}. The company builds fictional planning tools for museum puzzle exhibits.",
            )
            d2 = self.add_doc(
                p,
                f"{p} was born in {answer_city}. Before founding the company, the founder designed miniature transit diagrams.",
            )
            q = f"Which city is the birthplace of the founder of {company}?"
            a = answer_city
            meta = {"pattern": pattern, "anchor": company, "answer": a, "docs": [d1, d2]}

        elif pattern == "album_singer_hometown":
            album = title(case_index, "Album")
            d1 = self.add_doc(
                album,
                f"{album} is an album by singer {p}. Its songs describe invented markets, ferries, and hillside observatories.",
            )
            d2 = self.add_doc(
                p,
                f"{p} names {answer_city} as a hometown in the official artist profile. Several early performances happened near that city.",
            )
            q = f"What is the hometown of the singer behind {album}?"
            a = answer_city
            meta = {"pattern": pattern, "anchor": album, "answer": a, "docs": [d1, d2]}

        elif pattern == "discovery_university":
            effect = title(case_index, "Effect")
            d1 = self.add_doc(
                effect,
                f"The {effect} was first described by Dr. {p}. The effect concerns a synthetic shift in how colored dust scatters light.",
            )
            d2 = self.add_doc(
                p,
                f"{p} completed doctoral research at {inst}. The research notes later entered that university's teaching archive.",
            )
            q = f"At which university did the scientist who first described the {effect} complete doctoral research?"
            a = inst
            meta = {"pattern": pattern, "anchor": effect, "answer": a, "docs": [d1, d2]}

        elif pattern == "invention_institution":
            invention = title(case_index, "Device")
            d1 = self.add_doc(
                invention,
                f"The {invention} was invented by {p}. The device appears in this dataset as a fictional navigation instrument.",
            )
            d2 = self.add_doc(
                p,
                f"{p} was affiliated with {inst} during prototype development. The institution stored the design sketches in a classroom archive.",
            )
            q = f"Which institution was the inventor of the {invention} affiliated with during prototype development?"
            a = inst
            meta = {"pattern": pattern, "anchor": invention, "answer": a, "docs": [d1, d2]}

        elif pattern == "award_winner_employer":
            award = title(case_index, "Prize")
            d1 = self.add_doc(
                award,
                f"The 204{case_index % 10} {award} was won by {p}. The award recognizes synthetic field research in small observatories.",
            )
            d2 = self.add_doc(
                p,
                f"{p} works at {org}. The profile lists survey design and archive modeling as primary duties.",
            )
            q = f"Which organization employs the winner of the 204{case_index % 10} {award}?"
            a = org
            meta = {"pattern": pattern, "anchor": award, "answer": a, "docs": [d1, d2]}

        elif pattern == "mayor_birthplace":
            civic_city = f"{title(case_index, 'City').replace(' ', '')}"
            d1 = self.add_doc(
                f"{civic_city} Civic Record",
                f"The mayor of {civic_city} is {p}. The civic record is a synthetic entry used for entity-linking questions.",
            )
            d2 = self.add_doc(
                p,
                f"{p} was born in {answer_city}. The biography notes that the mayor later moved to several river districts.",
            )
            q = f"Which city is the birthplace of the mayor of {civic_city}?"
            a = answer_city
            meta = {"pattern": pattern, "anchor": civic_city, "answer": a, "docs": [d1, d2]}

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        self.add_qa(q, a, meta["docs"])
        self.generated_cases.append(meta)

    def add_three_hop_case(self, case_index: int, pattern: str) -> None:
        p = person(case_index)
        final_city = city(case_index + 17)
        inst = institution(case_index + 11)
        org = organization(case_index + 13)

        if pattern == "actor_company_headquarters":
            film = title(case_index, "Feature")
            company = title(case_index + 3, "Works").replace(" ", "")
            d1 = self.add_doc(
                film,
                f"{film} is a film starring {p}. The story centers on a survey crew crossing a fictional salt plain.",
            )
            d2 = self.add_doc(
                p,
                f"{p} founded {company} after leaving repertory theater. The company produces fictional stage-navigation exhibits.",
            )
            d3 = self.add_doc(
                company,
                f"{company} has its headquarters in {final_city}. Its public profile describes a small design office near the central arcade.",
            )
            q = f"Which city hosts the headquarters of the company founded by the actor who starred in {film}?"
            a = final_city
            docs = [d1, d2, d3]

        elif pattern == "scientist_award_administrator":
            signal = title(case_index, "Signal")
            award = title(case_index + 4, "Laurel")
            d1 = self.add_doc(
                signal,
                f"The {signal} was discovered by {p}. The signal is a synthetic astronomy phenomenon used in reasoning tests.",
            )
            d2 = self.add_doc(
                p,
                f"{p} won the {award} for cataloging fictional sky pulses. The award citation mentions the same signal study.",
            )
            d3 = self.add_doc(
                award,
                f"The {award} is administered by {inst}. The institution keeps a registry of all synthetic prize winners.",
            )
            q = f"Which institution administers the award won by the scientist who discovered the {signal}?"
            a = inst
            docs = [d1, d2, d3]

        elif pattern == "founder_university_city":
            company = title(case_index, "Aeroworks").replace(" ", "")
            university = institution(case_index + 5)
            d1 = self.add_doc(
                company,
                f"{company} was founded by {p}. The company designs fictional glider control kits for classrooms.",
            )
            d2 = self.add_doc(
                p,
                f"{p} attended {university}. The founder studied exhibit engineering there before forming the company.",
            )
            d3 = self.add_doc(
                university,
                f"{university} is located in {final_city}. The campus is described as standing beside a synthetic canal district.",
            )
            q = f"Which city is home to the university attended by the founder of {company}?"
            a = final_city
            docs = [d1, d2, d3]

        elif pattern == "author_award_sponsor":
            book = title(case_index, "Chronicle")
            award = title(case_index + 8, "Medal")
            d1 = self.add_doc(
                book,
                f"{book} is a novel written by {p}. The story follows archivists who sort invented weather tablets.",
            )
            d2 = self.add_doc(
                p,
                f"{p} won the {award} after publishing the novel. The citation praises compact fictional worldbuilding.",
            )
            d3 = self.add_doc(
                award,
                f"The {award} is sponsored by {org}. The organization funds synthetic literary fellowships.",
            )
            q = f"Which organization sponsors the award won by the author of {book}?"
            a = org
            docs = [d1, d2, d3]

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        self.add_qa(q, a, docs)

    def add_paraphrase_qa(self, case: dict, index: int) -> None:
        pattern = case["pattern"]
        anchor = case["anchor"]
        answer = case["answer"]
        docs = case["docs"]

        templates = {
            "book_birthplace": f"What city was the writer of {anchor} born in?",
            "film_actor_birthplace": f"What city was the star of {anchor} born in?",
            "company_founder_birthplace": f"What is the birth city of the founder of {anchor}?",
            "album_singer_hometown": f"Which hometown is listed for the singer of {anchor}?",
            "discovery_university": f"Where did the researcher behind the {anchor} complete doctoral research?",
            "invention_institution": f"What institution was associated with the inventor of the {anchor}?",
            "award_winner_employer": f"Where does the winner of the {anchor} work?",
            "mayor_birthplace": f"What city was the mayor linked to {anchor} born in?",
        }
        question = templates.get(pattern, f"What is the final attribute for {anchor}?")
        self.add_qa(question, answer, docs)

    def add_distractor_docs(self, count: int) -> None:
        for i in range(count):
            p = person(200 + i)
            distractor_title = title(180 + i, "Reference")
            self.add_doc(
                distractor_title,
                f"{distractor_title} mentions {p} and {city(140 + i)} as unrelated synthetic details. It is included as a distractor for retrieval tests.",
            )


def validate_dataset(docs: list[dict], qa_rows: list[dict]) -> None:
    required_doc_fields = {"doc_id", "title", "text", "source"}
    required_qa_fields = {"question", "answer", "supporting_doc_ids", "type"}
    doc_ids = set()

    for doc in docs:
        missing = required_doc_fields - set(doc)
        if missing:
            raise ValueError(f"Doc missing fields: {missing}")
        doc_id = doc["doc_id"]
        if doc_id in doc_ids:
            raise ValueError(f"Duplicate doc_id: {doc_id}")
        doc_ids.add(doc_id)

    for qa in qa_rows:
        missing = required_qa_fields - set(qa)
        if missing:
            raise ValueError(f"QA missing fields: {missing}")
        if len(qa["supporting_doc_ids"]) < 2:
            raise ValueError(f"QA needs at least two supporting docs: {qa}")
        for doc_id in qa["supporting_doc_ids"]:
            if doc_id not in doc_ids:
                raise ValueError(f"Unknown supporting doc_id {doc_id}: {qa['question']}")


def write_generation_prompt() -> None:
    prompt = """You are a synthetic dataset generator for a Search-Agent project.

Goal:
Create local retrieval documents and multi-hop QA examples for testing whether an agent can search, observe evidence, search again, and answer with citations.

Dataset requirements:
- docs.jsonl contains 200 synthetic local documents.
- qa.jsonl contains 100 synthetic multi-hop questions.
- Every QA answer must be supported by docs.jsonl.
- Every QA item must include supporting_doc_ids.
- Each question needs at least two reasoning hops.
- Answers must be short spans such as a city, person, year, institution, university, or organization.
- Use fictional people, books, films, companies, cities, awards, inventions, and institutions.
- Do not rely on real-world facts.
- Use strict JSONL with one valid JSON object per line.
- Use ASCII English text only.

Document schema:
{"doc_id":"doc001","title":"...","text":"...","source":"synthetic"}

QA schema:
{"question":"...","answer":"...","supporting_doc_ids":["doc001","doc002"],"type":"multi-hop"}

Reasoning patterns:
- book -> author -> birthplace
- film -> actor -> birthplace
- company -> founder -> birthplace
- album -> singer -> hometown
- discovery -> scientist -> university
- invention -> inventor -> institution
- award -> winner -> employer
- city -> mayor -> birthplace
- film -> actor -> company -> headquarters city
- discovery -> scientist -> award -> administering institution
"""
    PROMPT_PATH.write_text(prompt, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-docs", type=int, default=200)
    parser.add_argument("--target-qa", type=int, default=100)
    args = parser.parse_args()

    original_docs = load_jsonl(DOCS_PATH)
    original_qa = load_jsonl(QA_PATH)

    base_docs = [doc for doc in original_docs if doc_num(doc["doc_id"]) <= 50]
    base_qa = original_qa[:20]

    if len(base_docs) != 50:
        raise ValueError(f"Expected 50 base docs, found {len(base_docs)}")
    if len(base_qa) != 20:
        raise ValueError(f"Expected 20 base QA rows, found {len(base_qa)}")

    builder = DatasetBuilder(start_doc_id=51)
    patterns = [
        "book_birthplace",
        "film_actor_birthplace",
        "company_founder_birthplace",
        "album_singer_hometown",
        "discovery_university",
        "invention_institution",
        "award_winner_employer",
        "mayor_birthplace",
    ]

    for i in range(65):
        builder.add_two_hop_case(i, patterns[i % len(patterns)])

    three_hop_patterns = [
        "actor_company_headquarters",
        "scientist_award_administrator",
        "founder_university_city",
        "author_award_sponsor",
        "actor_company_headquarters",
    ]
    for offset, pattern in enumerate(three_hop_patterns, start=90):
        builder.add_three_hop_case(offset, pattern)

    for i, case in enumerate(builder.generated_cases[:10]):
        builder.add_paraphrase_qa(case, i)

    docs_needed = args.target_docs - len(base_docs) - len(builder.docs)
    if docs_needed < 0:
        raise ValueError("Generated more docs than target")
    builder.add_distractor_docs(docs_needed)

    docs = base_docs + builder.docs
    qa_rows = base_qa + builder.qa_rows

    if len(docs) != args.target_docs:
        raise ValueError(f"Expected {args.target_docs} docs, generated {len(docs)}")
    if len(qa_rows) != args.target_qa:
        raise ValueError(f"Expected {args.target_qa} QA rows, generated {len(qa_rows)}")

    validate_dataset(docs, qa_rows)
    write_jsonl(DOCS_PATH, docs)
    write_jsonl(QA_PATH, qa_rows)
    write_generation_prompt()

    print(
        json.dumps(
            {
                "docs_path": str(DOCS_PATH),
                "qa_path": str(QA_PATH),
                "doc_count": len(docs),
                "qa_count": len(qa_rows),
                "new_docs": len(builder.docs),
                "new_qa": len(builder.qa_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
