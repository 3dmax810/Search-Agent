import json
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.retriever.bm25 import BM25Retriever


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_bm25_can_load_docs():
    docs_path = root_path / "data" / "processed" / "docs.jsonl"
    retriever = BM25Retriever(docs_path)

    assert len(retriever.docs) == 50


def test_bm25_retrieves_som_supporting_doc_for_eval_questions():
    docs_path = root_path / "data" / "processed" / "docs.jsonl"
    qa_path = root_path / "data" / "eval" / "qa.jsonl"

    retriever = BM25Retriever(docs_path)
    qa_rows = load_jsonl(qa_path)

    hit_count = 0
    total = 0

    for qa in qa_rows:
        question = qa["question"]
        supporting_doc_ids = set(qa["supporting_doc_ids"])

        results = retriever.search(question, top_k=5)
        retrieved_doc_ids = {doc.doc_id for doc in results}

        if retrieved_doc_ids & supporting_doc_ids:
            hit_count += 1

        total += 1

    hit_rate = hit_count / total

    print(f"BM25 supporting-doc hit@5: {hit_count}/{total} = {hit_rate:.2%}")

    assert hit_rate >= 0.6
