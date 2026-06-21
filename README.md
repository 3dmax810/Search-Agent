# Search-Agent

A multi-turn search-augmented reasoning agent inspired by Search-R1. The agent can decide when to search, generate search queries, observe retrieved evidence, continue reasoning, and produce citation-grounded answers.

This project is not a plain RAG demo. It includes an Agent runtime, strict action parsing, local hybrid retrieval, answer validation, baseline evaluation, and case-study extraction.

## Motivation

Single-shot RAG often fails on multi-hop questions because the first retrieval step may only find an intermediate entity, not the final answer. Search-Agent addresses this by running a search-observe-answer loop:

```text
Question
  -> LLM emits <search> or <answer>
  -> Parser validates the action
  -> Search tool retrieves evidence
  -> Evidence is injected as <observe>
  -> LLM decides whether to search again or answer
```

For example, a question like:

```text
Which city is the birthplace of the author of The Silent Harbor?
```

requires two hops:

```text
The Silent Harbor -> Lena Moris -> Brookhaven
```

## Architecture

```text
User Question
    |
    v
AgentRuntime
    |
    v
Ollama LLM -> <search> query or <answer> final answer
    |
    v
Parser
    |
    v
LocalSearchTool
    |
    v
HybridRetriever = BM25 + Dense Retrieval + optional Reranker
    |
    v
<observe> evidence
    |
    v
Answer validation + AnswerJudge
    |
    v
Final cited answer
```

## Features

- Ollama-backed LLM agent runtime.
- XML-like action format: `<search>...</search>` and `<answer>...</answer>`.
- Parser-level format validation and fallback.
- Multi-turn search-observe-answer loop.
- Local hybrid retrieval with BM25 and FAISS dense retrieval.
- Local BGE embedding model support.
- Optional reranker interface.
- Evidence-based answer validation.
- LLM answer judge to reject intermediate-only answers.
- Baseline evaluation for no-search, single-shot RAG, and Search-Agent.

## Dataset

The current evaluation uses a synthetic multi-hop QA dataset.

```text
data/processed/docs.jsonl    200 local documents
data/eval/qa.jsonl           100 multi-hop QA samples
```

Example QA sample:

```json
{
  "question": "Which city is the birthplace of the author of The Silent Harbor?",
  "answer": "Brookhaven",
  "supporting_doc_ids": ["doc001", "doc002"],
  "type": "multi-hop"
}
```

The dataset was generated for controlled engineering validation. It is synthetic, so it should be treated as a development benchmark rather than a real-world benchmark. The first 50 documents and first 20 QA rows are the original smoke-test set; the remaining rows expand coverage across authors, actors, founders, singers, scientists, inventions, awards, mayors, and several three-hop chains.

## Retrieval

The retrieval stack has three layers:

```text
BM25Retriever
  Keyword-based retrieval using rank-bm25.

DenseRetriever
  Embedding-based retrieval using sentence-transformers and FAISS.

HybridRetriever
  Combines normalized BM25 and dense scores:
  score = 0.5 * bm25_norm + 0.5 * dense_norm
```

Supporting-document retrieval on the expanded 100-question synthetic set:

| Retriever | Supporting-doc Hit@5 |
|---|---:|
| BM25 | 99/100 = 99.00% |
| Dense | 20/100 = 20.00% |
| Hybrid | 100/100 = 100.00% |

The dense-only score above was recorded before switching the default embedding model to `bge-base-en`. Rebuild the FAISS index after changing the embedding model before comparing dense or hybrid results.

The current dense model path is:

```text
../models/Embedding/bge-base-en
```

The dense index is saved under:

```text
data/index/faiss.index
data/index/dense_docs.json
```

## Evaluation Results

Evaluation on the original 20-sample synthetic multi-hop QA benchmark:

| Method | EM | F1 | Citation Hit | Avg Search Turns | Avg Latency | Format Error Rate |
|---|---:|---:|---:|---:|---:|---:|
| No-search | 0.00 | 0.046 | 0.00 | 0.00 | 3.40s | 0.00 |
| Single-shot RAG | 0.10 | 0.135 | 0.65 | 1.00 | 3.74s | 0.00 |
| Search-Agent | 0.80 | 0.80 | 1.00 | 1.85 | 14.35s | 0.00 |

Search-Agent improves EM from `0.10` to `0.80` over single-shot RAG by using additional searches for intermediate entities and final attributes. The tradeoff is higher latency because the agent may call the LLM and retriever multiple times.

## Example Trace

Question:

```text
Which city is the birthplace of the author of The Silent Harbor?
```

Trace:

```text
<search>author of The Silent Harbor</search>
<observe>
[1] The Silent Harbor: The Silent Harbor is a novel written by Lena Moris.
</observe>

<search>Lena Moris birthplace</search>
<observe>
[1] Lena Moris: Lena Moris was born in Brookhaven.
</observe>

<answer>Brookhaven. [1]</answer>
```

## Case Studies

### No-search Fails, Search-Agent Succeeds

```text
Question:
Which city is the birthplace of the author of The Silent Harbor?

Gold answer:
Brookhaven

No-search prediction:
London

Single-shot RAG prediction:
Brookhaven [3]

Search-Agent prediction:
Brookhaven. [1]

Search-Agent search turns:
2
```

### Single-shot RAG Fails, Search-Agent Succeeds

```text
Question:
Which city is the birthplace of the author of Glass Meridian?

Gold answer:
Windmere

Single-shot RAG prediction:
No city is identified as the birthplace of the author of Glass Meridian in the provided evidence.

Search-Agent prediction:
Windmere. [1]

Search-Agent search turns:
2
```

### Current Failure Case

```text
Question:
Which city is the birthplace of the actor who starred in Echo Lantern?

Gold answer:
Northwick

Single-shot RAG prediction:
None of the provided evidence states the birthplace of the actor who starred in Echo Lantern.

Search-Agent prediction:
Joss Arden. [1]

Search-Agent search turns:
1
```

This failure shows a remaining weakness: the answer judge can still accept an intermediate entity when the final requested attribute is missing. A future improvement is a stronger answer-completeness judge or training data that teaches the model to always search the final attribute after identifying the intermediate entity.

## Quick Start

Build BM25 smoke report and dense FAISS index:

```powershell
python scripts\build_index.py --build-dense
```

Run the agent:

```powershell
python scripts\run_agent.py --question "Which city is the birthplace of the author of The Silent Harbor?"
```

Run evaluation:

```powershell
python scripts\run_eval.py --sample-size 20
```

Extract case studies:

```powershell
python scripts\extract_case_studies.py
```

Run tests:

```powershell
pytest tests -q
```

## Important Files

```text
search_agent/agent/runtime.py      Agent loop and trace recording
search_agent/agent/parser.py       <search>/<answer> parser
search_agent/agent/prompts.py      Agent prompt
search_agent/agent/judge.py        Answer judge
search_agent/retriever/bm25.py     BM25 retriever
search_agent/retriever/dense.py    FAISS dense retriever
search_agent/retriever/hybrid.py   BM25 + dense score fusion
search_agent/retriever/rerank.py   Reranker interface
search_agent/tools/search_tool.py  Search tool used by the agent
search_agent/eval/metrics.py       EM/F1/citation metrics
search_agent/eval/baselines.py     No-search, RAG, Search-Agent baselines
search_agent/eval/runners.py       Evaluation runner utilities
```

## Limitations

- The current dataset is synthetic and should be validated against a real-world benchmark before making broad claims.
- Dense and hybrid retrieval results depend on the embedding model used to build `faiss.index`; rebuild the index after switching models.
- The answer judge improves correctness but adds latency.
- Search-Agent is more accurate on multi-hop QA, but slower than single-shot RAG.
- The reranker is currently a placeholder and can be replaced with a cross-encoder such as `bge-reranker-base`.
- The current evaluation is development-oriented; larger HotpotQA or Wikipedia-based evaluation is needed for stronger claims.

## Next Steps

- Add a stronger answer-completeness judge.
- Replace the placeholder reranker with a real cross-encoder reranker.
- Re-run full evaluation on the expanded 100-question synthetic benchmark.
- Evaluate on HotpotQA or a larger Wikipedia-derived corpus.
- Add SFT traces to improve format following and multi-hop search behavior.


[
  {
    "method": "no-search",
    "count": 100,
    "em": 0.0,
    "f1": 0.023666666666666662,
    "citation_hit": 0.0,
    "avg_search_turns": 0.0,
    "avg_latency_seconds": 3.152682475000038,
    "format_error_rate": 0.0
  },
  {
    "method": "single-shot-rag",
    "count": 100,
    "em": 0.02,
    "f1": 0.027874873353596757,
    "citation_hit": 0.41,
    "avg_search_turns": 1.0,
    "avg_latency_seconds": 3.9160853659999337,
    "format_error_rate": 0.0
  },
  {
    "method": "search-agent",
    "count": 100,
    "em": 0.8,
    "f1": 0.8,
    "citation_hit": 0.89,
    "avg_search_turns": 2.18,
    "avg_latency_seconds": 16.120532632000195,
    "format_error_rate": 0.0
  }
]
