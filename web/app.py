from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
QA_PATH = ROOT / "data" / "eval" / "qa.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_agent.agent.runtime import AgentRuntime


st.set_page_config(
    page_title="Search-Agent Dashboard",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading Search-Agent...")
def get_agent(use_answer_judge: bool, use_memory: bool) -> AgentRuntime:
    from search_agent.agent.runtime import AgentRuntime

    return AgentRuntime(
        use_answer_judge=use_answer_judge,
        use_memory=use_memory,
    )


@st.cache_data(show_spinner=False)
def load_eval_summary() -> pd.DataFrame:
    path = RESULTS_DIR / "eval_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_case_studies() -> dict[str, list[dict[str, Any]]]:
    path = RESULTS_DIR / "case_studies.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_questions() -> list[str]:
    if not QA_PATH.exists():
        return []

    questions = []
    with QA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line)["question"])
    return questions


def format_number(value: Any, digits: int = 3) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def search_results_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, doc in enumerate(results, start=1):
        rows.append(
            {
                "rank": idx,
                "title": doc.get("title", ""),
                "score": round(float(doc.get("score", 0.0)), 4),
                "url": doc.get("url", ""),
                "snippet": doc.get("snippet", ""),
            }
        )
    return pd.DataFrame(rows)


def trace_stats(trace: list[dict[str, Any]]) -> dict[str, Any]:
    search_turns = 0
    format_errors = 0
    rejected_answers = 0

    for step in trace:
        if step.get("action") == "search":
            search_turns = max(search_turns, int(step.get("search_turns", 0)))
        if step.get("action") == "error":
            format_errors += 1
        if step.get("action") == "answer" and step.get("accepted") is False:
            rejected_answers += 1

    return {
        "turns": len(trace),
        "search_turns": search_turns,
        "format_errors": format_errors,
        "rejected_answers": rejected_answers,
    }


def show_trace(trace: list[dict[str, Any]], show_prompts: bool = False) -> None:
    if not trace:
        st.info("No trace available.")
        return

    for step in trace:
        turn = step.get("turn", "")
        action = step.get("action", "")
        content = step.get("content", "")
        accepted = step.get("accepted")

        suffix = ""
        if action == "search":
            suffix = f" | {content}"
        elif action == "answer" and accepted is not None:
            suffix = " | accepted" if accepted else " | rejected"
        elif action == "error":
            suffix = f" | {step.get('error', '')}"

        with st.expander(f"Turn {turn} | {action}{suffix}", expanded=False):
            cols = st.columns(4)
            cols[0].metric("search turns", step.get("search_turns", 0))
            cols[1].metric("max search", step.get("max_search_turns", 0))
            cols[2].metric("context chars", step.get("context_chars", 0))
            cols[3].metric("prompt chars", step.get("prompt_chars", 0))

            if step.get("error"):
                st.error(step["error"])
            if step.get("reject_reason"):
                st.warning(step["reject_reason"])
            if step.get("judge_reason"):
                st.info(step["judge_reason"])

            if content:
                st.markdown("**Parsed content**")
                st.code(content, language="text")

            if step.get("model_output"):
                st.markdown("**Model output**")
                st.code(step["model_output"], language="xml")

            results = step.get("search_results", [])
            if results:
                st.markdown("**Search results**")
                st.dataframe(
                    search_results_frame(results),
                    hide_index=True,
                    width="stretch",
                )

            if show_prompts and step.get("prompt"):
                st.markdown("**Prompt**")
                st.code(step["prompt"], language="text")


def parse_raw_agent_output(raw_output: str) -> dict[str, Any] | None:
    if not raw_output:
        return None
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return None


def show_eval_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.warning("No eval_summary.csv found.")
        return

    search_agent_rows = summary[summary["method"] == "search-agent"]
    if not search_agent_rows.empty:
        row = search_agent_rows.iloc[0]
        cols = st.columns(5)
        cols[0].metric("EM", format_number(row["em"]))
        cols[1].metric("F1", format_number(row["f1"]))
        cols[2].metric("Citation hit", format_number(row["citation_hit"]))
        cols[3].metric("Avg searches", format_number(row["avg_search_turns"]))
        cols[4].metric("Avg latency", f"{row['avg_latency_seconds']:.2f}s")

    st.dataframe(summary, hide_index=True, width="stretch")


def show_case_studies(case_studies: dict[str, list[dict[str, Any]]]) -> None:
    if not case_studies:
        st.warning("No case_studies.json found.")
        return

    categories = list(case_studies.keys())
    category = st.selectbox("Category", categories)
    cases = case_studies.get(category, [])

    if not cases:
        st.info("No cases in this category.")
        return

    for idx, case in enumerate(cases, start=1):
        title = case.get("question", f"Case {idx}")
        with st.expander(title, expanded=idx == 1):
            cols = st.columns(4)
            cols[0].metric("gold", case.get("gold_answer", ""))
            cols[1].metric("no-search", case.get("no_search_prediction", ""))
            cols[2].metric("single-shot", case.get("single_shot_prediction", ""))
            cols[3].metric("search turns", case.get("search_agent_search_turns", 0))

            st.markdown("**Search-Agent prediction**")
            st.code(case.get("search_agent_prediction", ""), language="text")

            parsed = parse_raw_agent_output(case.get("search_agent_raw_output", ""))
            if parsed:
                show_trace(parsed.get("trace", []), show_prompts=False)


def run_agent_panel() -> None:
    questions = load_questions()
    default_question = (
        questions[0]
        if questions
        else "Which city is the birthplace of the author of The Silent Harbor?"
    )

    with st.sidebar:
        use_answer_judge = st.toggle("Answer judge", value=True)
        use_memory = st.toggle("Memory", value=True)
        show_prompts = st.toggle("Show prompts", value=False)
        selected_question = st.selectbox(
            "Example question",
            questions or [default_question],
        )

    question = st.text_area(
        "Question",
        value=selected_question,
        height=90,
    )

    if st.button("Run Search-Agent", type="primary"):
        agent = get_agent(
            use_answer_judge=use_answer_judge,
            use_memory=use_memory,
        )
        with st.spinner("Running multi-turn search..."):
            st.session_state["agent_result"] = agent.run(question)

    result = st.session_state.get("agent_result")
    if not result:
        return

    answer = result.get("answer", "")
    trace = result.get("trace", [])
    stats = trace_stats(trace)

    st.markdown("**Final answer**")
    st.success(answer)

    memory_used = any(step.get("memory_used") for step in trace)

    cols = st.columns(5)
    cols[0].metric("turns", stats["turns"])
    cols[1].metric("search turns", stats["search_turns"])
    cols[2].metric("format errors", stats["format_errors"])
    cols[3].metric("rejected answers", stats["rejected_answers"])
    cols[4].metric("memory", "on" if memory_used else "off")

    st.markdown("**Trace**")
    show_trace(trace, show_prompts=show_prompts)


def main() -> None:
    st.title("Search-Agent Dashboard")

    tab_run, tab_eval, tab_cases = st.tabs(
        ["Agent Trace", "Evaluation", "Case Studies"]
    )

    with tab_run:
        run_agent_panel()

    with tab_eval:
        show_eval_summary(load_eval_summary())

    with tab_cases:
        show_case_studies(load_case_studies())


if __name__ == "__main__":
    main()
