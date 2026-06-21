import json

from scripts.prepare_musique import convert_musique_result


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_prepare_musique_dedupes_docs_and_keeps_support_order(tmp_path):
    input_path = tmp_path / "musique.jsonl"
    rows = [
        {
            "id": "sample_a",
            "answerable": True,
            "paragraphs": [
                {
                    "idx": 2,
                    "title": "Answer Entity",
                    "paragraph_text": "The answer entity was founded by Ada.",
                    "is_supporting": True,
                },
                {
                    "idx": 1,
                    "title": "Bridge Entity",
                    "paragraph_text": "The bridge entity points to the answer entity.",
                    "is_supporting": True,
                },
                {
                    "idx": 3,
                    "title": "Shared Distractor",
                    "paragraph_text": "This paragraph appears in more than one sample.",
                    "is_supporting": False,
                },
            ],
            "question": "Who founded the answer entity linked by the bridge entity?",
            "question_decomposition": [
                {
                    "id": 1,
                    "question": "bridge entity >> linked entity",
                    "answer": "Answer Entity",
                    "paragraph_support_idx": 1,
                },
                {
                    "id": 2,
                    "question": "#1 >> founded by",
                    "answer": "Ada",
                    "paragraph_support_idx": 2,
                },
            ],
            "answer": "Ada",
            "answer_aliases": [],
        },
        {
            "id": "sample_b",
            "answerable": True,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Shared Distractor",
                    "paragraph_text": "This paragraph appears in more than one sample.",
                    "is_supporting": False,
                },
                {
                    "idx": 1,
                    "title": "Other Support",
                    "paragraph_text": "Other support says the answer is Grace.",
                    "is_supporting": True,
                },
            ],
            "question": "Who is named by other support?",
            "question_decomposition": [
                {
                    "id": 3,
                    "question": "other support >> named person",
                    "answer": "Grace",
                    "paragraph_support_idx": 1,
                }
            ],
            "answer": "Grace",
            "answer_aliases": [],
        },
    ]
    write_jsonl(input_path, rows)

    result = convert_musique_result(input_path=input_path)

    assert result.stats["dedupe"] is True
    assert result.stats["paragraph_occurrences"] == 5
    assert result.stats["duplicate_paragraph_occurrences"] == 1
    assert len(result.docs) == 4

    shared_docs = [doc for doc in result.docs if doc["title"] == "Shared Distractor"]
    assert len(shared_docs) == 1
    assert shared_docs[0]["source_ref_count"] == 2

    first_qa = result.qa_rows[0]
    assert first_qa["original_supporting_refs"][0]["paragraph_idx"] == 1
    assert first_qa["original_supporting_refs"][1]["paragraph_idx"] == 2
    assert first_qa["supporting_doc_ids"] == [
        first_qa["original_supporting_refs"][0]["doc_id"],
        first_qa["original_supporting_refs"][1]["doc_id"],
    ]
