import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_question(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["question"]][row["method"]] = row
    return grouped


def compact_case(question: str, rows_by_method: dict[str, dict]) -> dict:
    no_search = rows_by_method.get("no-search", {})
    single_shot = rows_by_method.get("single-shot-rag", {})
    search_agent = rows_by_method.get("search-agent", {})

    return {
        "question": question,
        "gold_answer": (
            search_agent.get("gold_answer")
            or single_shot.get("gold_answer")
            or no_search.get("gold_answer")
        ),
        "no_search_prediction": no_search.get("prediction", ""),
        "single_shot_prediction": single_shot.get("prediction", ""),
        "search_agent_prediction": search_agent.get("prediction", ""),
        "search_agent_search_turns": search_agent.get("search_turns", None),
        "search_agent_raw_output": search_agent.get("raw_output", ""),
    }


def select_cases(grouped: dict[str, dict[str, dict]]) -> dict[str, list[dict]]:
    cases = {
        "no_search_wrong_search_agent_correct": [],
        "single_shot_wrong_search_agent_correct": [],
        "search_agent_failed": [],
    }

    for question, rows_by_method in grouped.items():
        no_search = rows_by_method.get("no-search")
        single_shot = rows_by_method.get("single-shot-rag")
        search_agent = rows_by_method.get("search-agent")

        if not search_agent:
            continue

        search_agent_correct = search_agent.get("em", 0.0) == 1.0
        no_search_correct = no_search and no_search.get("em", 0.0) == 1.0
        single_shot_correct = single_shot and single_shot.get("em", 0.0) == 1.0

        case = compact_case(question, rows_by_method)

        if no_search and (not no_search_correct) and search_agent_correct:
            cases["no_search_wrong_search_agent_correct"].append(case)

        if single_shot and (not single_shot_correct) and search_agent_correct:
            cases["single_shot_wrong_search_agent_correct"].append(case)

        if not search_agent_correct:
            cases["search_agent_failed"].append(case)

    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--details-path",
        default=str(root_path / "results" / "eval_details.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        default=str(root_path / "results" / "case_studies.json"),
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.details_path))
    grouped = group_by_question(rows)
    cases = select_cases(grouped)

    trimmed_cases = {
        case_type: items[: args.max_per_type] for case_type, items in cases.items()
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(trimmed_cases, f, ensure_ascii=False, indent=2)

    print(json.dumps(trimmed_cases, ensure_ascii=False, indent=2))
    print(f"Wrote case studies: {output_path}")


if __name__ == "__main__":
    main()
