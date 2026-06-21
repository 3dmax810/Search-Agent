import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ConversionResult:
    docs: list[dict]
    qa_rows: list[dict]
    stats: dict


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_id(value: str) -> str:
    safe_value = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")
    return safe_value or "unknown"


def normalize_content(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def content_hash(title: str, text: str) -> str:
    key = f"{normalize_content(title)}\n{normalize_content(text)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def canonical_doc_id(title: str, text: str) -> str:
    return f"musique_doc_{content_hash(title, text)}"


def paragraph_doc_id(sample_id: str, paragraph_idx: int) -> str:
    return f"musique_{safe_id(sample_id)}_p{paragraph_idx}"


def make_source_ref(sample_id: str, paragraph_idx: int, is_supporting: bool) -> dict:
    return {
        "sample_id": sample_id,
        "paragraph_idx": paragraph_idx,
        "is_supporting": is_supporting,
    }


def make_supporting_ref(
    sample_id: str,
    paragraph_idx: int,
    doc_id: str,
    title: str,
    text: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "paragraph_idx": paragraph_idx,
        "doc_id": doc_id,
        "title": title,
        "content_hash": content_hash(title, text),
    }


def append_unique_dict(rows: list[dict], row: dict) -> None:
    if row not in rows:
        rows.append(row)


def append_unique_value(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def build_doc_row(
    doc_id: str,
    title: str,
    text: str,
    source: str,
    source_refs: list[dict],
) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "source": source,
        "chunk_type": "paragraph",
        "content_hash": content_hash(title, text),
        "source_refs": source_refs,
        "source_ref_count": len(source_refs),
    }


def support_order_from_decomposition(item: dict, paragraph_order: list[int]) -> list[int]:
    ordered_idxs: list[int] = []

    for step in item.get("question_decomposition", []) or []:
        support_idx = step.get("paragraph_support_idx")
        if isinstance(support_idx, int) and support_idx not in ordered_idxs:
            ordered_idxs.append(support_idx)

    for paragraph in item.get("paragraphs", []) or []:
        paragraph_idx = paragraph.get("idx")
        if paragraph.get("is_supporting") and paragraph_idx not in ordered_idxs:
            ordered_idxs.append(paragraph_idx)

    valid_idxs = set(paragraph_order)
    return [idx for idx in ordered_idxs if idx in valid_idxs]


def convert_musique_result(
    input_path: Path,
    sample_size: int | None = None,
    source: str = "musique",
    dedupe: bool = True,
) -> ConversionResult:
    docs: list[dict] = []
    qa_rows: list[dict] = []
    canonical_docs: dict[str, dict] = {}
    seen_doc_ids: set[str] = set()

    stats = {
        "input_rows": 0,
        "answerable_rows": 0,
        "used_rows": 0,
        "skipped_unanswerable": 0,
        "skipped_no_support": 0,
        "paragraph_occurrences": 0,
        "duplicate_paragraph_occurrences": 0,
        "dedupe": dedupe,
    }

    for item in iter_jsonl(input_path):
        stats["input_rows"] += 1

        if item.get("answerable") is False:
            stats["skipped_unanswerable"] += 1
            continue

        stats["answerable_rows"] += 1
        sample_id = str(item["id"])
        paragraphs = item.get("paragraphs", []) or []

        doc_id_by_idx: dict[int, str] = {}
        paragraph_by_idx: dict[int, dict] = {}
        paragraph_order: list[int] = []

        for paragraph in paragraphs:
            paragraph_idx = int(paragraph["idx"])
            title = paragraph.get("title", "")
            text = paragraph.get("paragraph_text", "")
            is_supporting = bool(paragraph.get("is_supporting"))
            source_ref = make_source_ref(sample_id, paragraph_idx, is_supporting)

            stats["paragraph_occurrences"] += 1
            paragraph_order.append(paragraph_idx)
            paragraph_by_idx[paragraph_idx] = {
                "title": title,
                "text": text,
                "is_supporting": is_supporting,
            }

            if dedupe:
                doc_id = canonical_doc_id(title, text)
                if doc_id not in canonical_docs:
                    canonical_docs[doc_id] = build_doc_row(
                        doc_id=doc_id,
                        title=title,
                        text=text,
                        source=source,
                        source_refs=[],
                    )
                else:
                    stats["duplicate_paragraph_occurrences"] += 1

                append_unique_dict(canonical_docs[doc_id]["source_refs"], source_ref)
                canonical_docs[doc_id]["source_ref_count"] = len(
                    canonical_docs[doc_id]["source_refs"]
                )
            else:
                doc_id = paragraph_doc_id(sample_id, paragraph_idx)
                if doc_id in seen_doc_ids:
                    raise ValueError(f"Duplicate generated doc_id: {doc_id}")

                seen_doc_ids.add(doc_id)
                docs.append(
                    build_doc_row(
                        doc_id=doc_id,
                        title=title,
                        text=text,
                        source=source,
                        source_refs=[source_ref],
                    )
                )

            doc_id_by_idx[paragraph_idx] = doc_id

        support_idxs = support_order_from_decomposition(item, paragraph_order)
        if not support_idxs:
            stats["skipped_no_support"] += 1
            continue

        supporting_doc_ids: list[str] = []
        original_supporting_refs: list[dict] = []

        for paragraph_idx in support_idxs:
            doc_id = doc_id_by_idx[paragraph_idx]
            paragraph = paragraph_by_idx[paragraph_idx]
            append_unique_value(supporting_doc_ids, doc_id)
            original_supporting_refs.append(
                make_supporting_ref(
                    sample_id=sample_id,
                    paragraph_idx=paragraph_idx,
                    doc_id=doc_id,
                    title=paragraph["title"],
                    text=paragraph["text"],
                )
            )

        qa_rows.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "answer_aliases": item.get("answer_aliases", []),
                "supporting_doc_ids": supporting_doc_ids,
                "original_supporting_refs": original_supporting_refs,
                "question_decomposition": item.get("question_decomposition", []),
                "type": "multi-hop",
                "source": source,
                "source_id": sample_id,
                "hop_count": len(item.get("question_decomposition", []) or []),
            }
        )

        stats["used_rows"] += 1
        if sample_size is not None and stats["used_rows"] >= sample_size:
            break

    if dedupe:
        docs = list(canonical_docs.values())

    stats["doc_count"] = len(docs)
    stats["qa_count"] = len(qa_rows)
    return ConversionResult(docs=docs, qa_rows=qa_rows, stats=stats)


def convert_musique(
    input_path: Path,
    sample_size: int | None = None,
    source: str = "musique",
    dedupe: bool = True,
) -> tuple[list[dict], list[dict]]:
    result = convert_musique_result(
        input_path=input_path,
        sample_size=sample_size,
        source=source,
        dedupe=dedupe,
    )
    return result.docs, result.qa_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MuSiQue JSONL into Search-Agent docs.jsonl and qa.jsonl."
    )
    parser.add_argument(
        "--input",
        default="Githubs/musique/musique_ans_v1.0_dev.jsonl",
        help="Path to a MuSiQue JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/musique",
        help="Directory for converted docs.jsonl and qa.jsonl.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of answerable MuSiQue examples to convert. Use 0 for full file.",
    )
    parser.add_argument(
        "--source",
        default="musique",
        help="Source label written into docs and QA rows.",
    )
    parser.add_argument(
        "--dedupe",
        dest="dedupe",
        action="store_true",
        default=True,
        help="Deduplicate paragraphs by normalized title + paragraph text. Default: on.",
    )
    parser.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="Use sample-specific paragraph doc_ids instead of global canonical doc_ids.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"MuSiQue input file not found: {input_path}")

    result = convert_musique_result(
        input_path=input_path,
        sample_size=None if args.sample_size == 0 else args.sample_size,
        source=args.source,
        dedupe=args.dedupe,
    )

    write_jsonl(output_dir / "docs.jsonl", result.docs)
    write_jsonl(output_dir / "qa.jsonl", result.qa_rows)

    report = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "sample_size": args.sample_size,
        **result.stats,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
