import json
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.retriever.dense import DenseRetriever


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_dense_retriever() -> DenseRetriever:
    return DenseRetriever(
        docs_path=root_path / "data" / "processed" / "docs.jsonl",
        model_path=root_path.parent / "models" / "Embedding" / "bge-base-zh",
        index_path=root_path / "data" / "index" / "faiss.index",
        meta_path=root_path / "data" / "index" / "dense_docs.json",
    )


def test_dense_index_files_exist():
    assert (root_path / "data" / "index" / "faiss.index").exists()
    assert (root_path / "data" / "index" / "dense_docs.json").exists()


def test_dense_retriever_can_search():
    retriever = make_dense_retriever()
    results = retriever.search("author of The Silent Harbor", top_k=3)

    assert len(results) == 3
    assert "doc001" in {doc.doc_id for doc in results}


def test_dense_retrieves_some_supporting_doc_for_eval_questions():
    qa_path = root_path / "data" / "eval" / "qa.jsonl"
    qa_rows = load_jsonl(qa_path)

    retriever = make_dense_retriever()

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

    print(f"Dense supporting-doc hit@5: {hit_count}/{total} = {hit_rate:.2%}")
    assert hit_rate >= 0.5
