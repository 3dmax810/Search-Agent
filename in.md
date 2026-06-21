# MuSiQue 检索召回实验记录

## 实验设置

数据集：MuSiQue answerable dev

评测样本数：2417

文档数：48315

检索方式：

- BM25：关键词稀疏检索
- Dense：向量语义检索
- Hybrid：BM25 与 Dense 分数融合

## 指标说明

| 指标 | 含义 | 对多跳任务的意义 |
| --- | --- | --- |
| `hit@k` | Top-k 结果中是否至少命中 1 个支持文档 | 判断检索器是否能找到任意一跳证据 |
| `all_support_hit@k` | Top-k 结果中是否命中全部支持文档 | 判断一次检索是否能找齐完整多跳证据 |
| `support_recall@k` | Top-k 结果中召回的支持文档比例 | 衡量多跳证据整体召回能力 |
| `mrr@k` | 第一个支持文档出现位置的倒数排名 | 衡量支持证据是否靠前 |

## Top-1 对比

| Retriever | Hit@1 | All Support Hit@1 | Support Recall@1 | MRR@1 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 33.88% | 0.00% | 14.84% | 33.88% |
| Hybrid | 34.92% | 0.00% | 15.39% | 34.92% |
| Dense | 18.25% | 0.00% | 8.29% | 18.25% |

## Top-3 对比

| Retriever | Hit@3 | All Support Hit@3 | Support Recall@3 | MRR@3 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 50.56% | 1.37% | 22.53% | 41.25% |
| Hybrid | 50.89% | 0.87% | 22.33% | 41.95% |
| Dense | 28.67% | 0.33% | 12.91% | 22.87% |

## Top-5 对比

| Retriever | Hit@5 | All Support Hit@5 | Support Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 57.34% | 3.52% | 26.60% | 42.82% |
| Hybrid | 57.67% | 2.52% | 26.05% | 43.49% |
| Dense | 34.63% | 1.16% | 15.89% | 24.23% |

## Top-10 对比

| Retriever | Hit@10 | All Support Hit@10 | Support Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 65.25% | 6.91% | 32.04% | 43.89% |
| Hybrid | 66.03% | 5.96% | 31.53% | 44.61% |
| Dense | 43.24% | 2.23% | 19.95% | 25.34% |

## 结果分析

### 1. 单次检索很难解决 MuSiQue 多跳问题

BM25 的 `hit@10` 达到 65.25%，说明 Top-10 中经常能找到至少一个支持文档。

但是 BM25 的 `all_support_hit@10` 只有 6.91%，`support_recall@10` 只有 32.04%。这说明单次检索通常只能找到部分证据，难以一次性找齐多跳问题所需的完整证据。

这正好支持 Search-Agent 的设计动机：

```text
Single-shot RAG 一次检索证据不完整
Search-Agent 通过多轮 search -> observe -> search 逐跳补齐证据
```

### 2. Hybrid 目前只略微提升 Hit 和 MRR

Hybrid 的 `hit@10` 是 66.03%，略高于 BM25 的 65.25%。

Hybrid 的 `mrr@10` 是 44.61%，也略高于 BM25 的 43.89%。

但是 Hybrid 的 `all_support_hit@10` 和 `support_recall@10` 都低于 BM25：

| Metric | BM25 | Hybrid |
| --- | ---: | ---: |
| All Support Hit@10 | 6.91% | 5.96% |
| Support Recall@10 | 32.04% | 31.53% |

这说明当前 Hybrid 融合没有明显改善多跳证据的完整召回。

### 3. Dense 效果明显弱于 BM25

Dense 的 `hit@10` 只有 43.24%，明显低于 BM25 的 65.25%。

主要原因可能是这次实验构建 Dense 索引时使用的是 `bge-base-zh`，而 MuSiQue 是英文数据集。中文 embedding 模型在英文 Wikipedia 段落上的语义匹配能力有限。

当前项目默认 embedding 模型已切换为 `bge-base-en`。切换模型后必须重新构建 FAISS 索引，再重新评测 Dense 和 Hybrid。可选模型包括：

```text
bge-base-en
bge-large-en
bge-m3
e5-base-v2
```

### 4. 当前最有价值的结论

这组三组结果可以用于项目说明：

```text
在 MuSiQue answerable dev 全量 2417 条样本上，BM25 单次检索 hit@10 为 65.25%，但 all-support-hit@10 仅 6.91%，说明单轮 RAG 难以一次召回多跳问题所需的完整证据。因此项目引入 Search-Agent，通过多轮查询生成和证据观察逐步补齐推理链。
```

## 下一步操作

### 第一步：先不要继续调 Dense

当前 Dense 效果较弱，主要受 embedding 模型影响。短期内先不要把时间花在 Dense 权重调参上。

当前推荐先使用 BM25 或 Hybrid 作为 Agent 检索工具。

### 第二步：跑 MuSiQue 单条 Agent 多跳样例

先验证 Agent 是否能在真实 MuSiQue 问题上走出多跳链路：

```powershell
python scripts\run_agent.py `
  --question "Who is the spouse of the Green performer?" `
  --docs-path data\musique\docs.jsonl `
  --index-dir data\musique\index `
  --no-memory
```

期望轨迹：

```text
search: Green performer
observe: Green -> Steve Hillage
search: Steve Hillage spouse
observe: Miquette Giraudy
answer: Miquette Giraudy [citation]
```

### 第三步：跑小样本 Agent 评测

单条样例通过后，再跑 10 条样本：

```powershell
python scripts\run_eval.py `
  --qa-path data\musique\qa.jsonl `
  --docs-path data\musique\docs.jsonl `
  --index-dir data\musique\index `
  --sample-size 10 `
  --methods single-shot-rag search-agent
```

目标是观察：

```text
search-agent 是否优于 single-shot-rag
平均 search_turns 是否接近 2
失败案例主要来自检索失败还是模型没有继续搜索
```

### 第四步：如果 Agent 成功率低，优先调这三个点

1. 提高每次搜索返回的 `top_k`
2. 强化 prompt 中“找到中间实体后必须搜索目标属性”的规则
3. 在 AnswerJudge 中拒绝只回答中间实体的答案

### 第五步：后续增强方向

在 MuSiQue 小样本 Agent 评测跑通后，再做工程增强：

- 增加 `ToolRegistry`，为后续 MCP 工具预留统一接口
- 增加 `MemoryManager`，把当前记忆扩展成多层记忆入口
- 增加失败案例分析脚本，统计检索失败、格式失败、答案不完整等原因
- 在 Streamlit dashboard 中展示检索召回表格和 Agent trace

---

# Search-Agent 项目整体内容补充

## 项目定位

Search-Agent 不是普通的单轮 RAG Demo，而是一个多轮搜索增强推理 Agent 工程。

项目目标是让 LLM 在回答复杂问题时，可以自主决定：

```text
是否需要搜索
搜索什么 query
如何根据搜索结果继续推理
什么时候输出最终答案
```

核心思想可以概括为：

```text
LLM 负责生成动作
Agent Harness 负责约束、执行、校验、记录和评测
```

因此项目重点不是单纯调用大模型，而是围绕大模型构建一套可控的 Agent Runtime。

## 当前系统架构

当前系统的主链路如下：

```text
User Question
  -> AgentRuntime
  -> OllamaModel
  -> Parser
  -> LocalSearchTool
  -> HybridRetriever
  -> BM25Retriever / DenseRetriever
  -> <observe> evidence
  -> Answer validation
  -> AnswerJudge
  -> Final answer with citation
```

核心文件：

```text
search_agent/agent/runtime.py       Agent 主循环
search_agent/agent/parser.py        解析 LLM 输出动作
search_agent/agent/prompts.py       构造 Agent prompt
search_agent/agent/context.py       上下文组装与截断
search_agent/agent/judge.py         答案裁判
search_agent/tools/search_tool.py   搜索工具封装
search_agent/retriever/bm25.py      BM25 稀疏检索
search_agent/retriever/dense.py     FAISS 向量检索
search_agent/retriever/hybrid.py    BM25 + Dense 融合检索
search_agent/memory/store.py        经验记忆存储
search_agent/eval/runners.py        评测执行逻辑
search_agent/eval/baselines.py      No-search / RAG / Search-Agent baseline
```

## Agent 运行流程

Agent 每一轮只允许输出一个动作：

```text
<search>...</search>
```

或：

```text
<answer>...</answer>
```

运行过程：

```text
1. 用户输入问题
2. ContextAssembler 组装 prompt
3. LLM 输出 search 或 answer
4. Parser 校验格式
5. 如果是 search，调用 LocalSearchTool
6. 检索结果被写成 <observe>
7. LLM 根据 observe 决定继续搜索或回答
8. 如果是 answer，系统检查 citation 和 evidence support
9. AnswerJudge 二次判断是否回答了原始问题
10. 合格则返回答案，不合格则拒绝并要求继续搜索
```

这使得 Agent 可以完成多跳搜索，例如：

```text
问题：Which city is the birthplace of the author of The Silent Harbor?

第 1 跳：
search: author of The Silent Harbor
observe: The Silent Harbor is written by Lena Moris

第 2 跳：
search: Lena Moris birthplace
observe: Lena Moris was born in Brookhaven

最终答案：
Brookhaven [citation]
```

## Harness 工程内容

当前项目已经包含以下 Harness 工程能力：

```text
1. 有界状态机
   - max_model_turns
   - max_search_turns
   - empty output retry
   - search limit halt

2. 格式约束
   - 只允许 <search> 或 <answer>
   - 禁止同轮同时 search 和 answer
   - 禁止标签外文本
   - answer 必须带 citation

3. 证据约束
   - 没有 observe 不允许 answer
   - answer 必须被 observe 支持
   - evidence insufficient 时继续 search

4. AnswerJudge
   - 拒绝只回答中间实体的答案
   - 判断答案是否真正回答了原问题

5. Trace 可复盘
   - 每轮保存 prompt
   - 保存 model_output
   - 保存 action/content/error
   - 保存 search_results
   - 保存 accepted/reject_reason/judge_reason

6. Context 工程
   - 上下文统一组装
   - trace 截断
   - prompt_chars / context_chars 记录
   - memory_context 注入

7. Memory 雏形
   - 成功轨迹写入 MemoryStore
   - 相似问题可检索历史经验
   - 记忆只辅助规划，不作为最终证据
```

## 当前已实现能力

已经实现：

```text
Agent Runtime
严格动作解析
BM25 检索
Dense FAISS 检索
Hybrid 检索
Ollama 本地 LLM 接入
AnswerJudge
ContextAssembler
MemoryStore
No-search baseline
Single-shot RAG baseline
Search-Agent baseline
EM / F1 / citation_hit 评测
检索召回评测 hit@k / recall@k / MRR@k
Streamlit dashboard
MuSiQue 数据转换脚本
MuSiQue 检索召回实验
```

尚未完全实现或后续增强：

```text
ToolRegistry
MCP 工具封装
多层记忆 MemoryManager
FastAPI 服务层
更完整的失败案例自动归因
英文或多语言 embedding 模型替换实验
Reranker 实验
Agent 在 MuSiQue 上的全量评测
```

## 数据集结构

当前项目支持两类数据：

```text
docs.jsonl  文档库，用于检索
qa.jsonl    问题集，用于评测
```

`docs.jsonl` 示例：

```json
{
  "doc_id": "doc001",
  "title": "The Silent Harbor",
  "text": "The Silent Harbor is a novel written by Lena Moris.",
  "source": "synthetic"
}
```

`qa.jsonl` 示例：

```json
{
  "question": "Which city is the birthplace of the author of The Silent Harbor?",
  "answer": "Brookhaven",
  "supporting_doc_ids": ["doc001", "doc002"],
  "type": "multi-hop"
}
```

MuSiQue 原始文件中，一条样本同时包含：

```text
question
answer
paragraphs
question_decomposition
answer_aliases
answerable
```

转换后：

```text
paragraphs -> docs.jsonl
question/answer/supporting_doc_ids -> qa.jsonl
```

`question_decomposition` 只用于分析，不注入 Agent prompt，避免泄露推理路径。

## 评测体系

当前评测分为两层：

### 1. 检索评测

脚本：

```text
scripts/eval_retrieval.py
```

指标：

```text
hit@k
all_support_hit@k
support_recall@k
mrr@k
```

用途：

```text
判断检索器是否能召回支持证据
判断单次 RAG 是否有机会回答完整多跳问题
比较 BM25 / Dense / Hybrid 的差异
```

### 2. Agent 评测

脚本：

```text
scripts/run_eval.py
```

方法：

```text
no-search
single-shot-rag
search-agent
```

指标：

```text
EM
F1
citation_hit
avg_search_turns
avg_latency_seconds
format_error_rate
```

用途：

```text
验证 Search-Agent 是否优于无搜索和单轮 RAG
观察多轮搜索是否能补齐证据链
分析准确率与延迟之间的权衡
```

## 当前 MuSiQue 实验结论

MuSiQue answerable dev 全量：

```text
样本数：2417
文档数：48315
```

当前检索结果说明：

```text
BM25 hit@10 = 65.25%
BM25 all_support_hit@10 = 6.91%
BM25 support_recall@10 = 32.04%
```

结论：

```text
单轮检索经常能找到一个相关证据，但很难一次找齐所有多跳支持证据。
这说明 single-shot RAG 在 MuSiQue 上天然受限。
Search-Agent 的多轮 search-observe-search 机制有明确必要性。
```

Dense 当前弱于 BM25，主要原因是：

```text
这组历史结果构建索引时使用 bge-base-zh
MuSiQue 是英文 Wikipedia 数据
中文 embedding 模型对英文语义匹配能力有限
当前默认模型已切换为 bge-base-en，需要重建索引并重新评测
```

后续可替换为：

```text
bge-base-en
bge-large-en
bge-m3
e5-base-v2
```

## 面向大模型开发岗的项目亮点

可以在简历或面试中表达为：

```text
独立实现一个多轮搜索增强推理 Agent，围绕 LLM 构建可约束、可终止、可复盘、可评测的 Agent Harness。
系统支持 XML 动作解析、BM25 + FAISS 混合检索、证据约束回答、AnswerJudge 二次校验、经验记忆、上下文截断和多 baseline 评测。
在 MuSiQue 多跳问答数据集上构建检索召回实验，验证单轮 RAG 难以一次召回完整多跳证据，并为 Search-Agent 的多轮检索设计提供实验依据。
```

STAR 版本：

```text
S：多跳问答中，单轮 RAG 往往只能检索到中间实体，难以召回完整证据链。
T：设计一个可控的 Search-Agent，使 LLM 能自主生成查询、多轮调用检索工具，并基于证据输出答案。
A：实现 Agent Runtime、严格动作解析、Hybrid Retriever、AnswerJudge、MemoryStore 和统一评测框架；接入 MuSiQue 构建真实多跳检索实验。
R：在 MuSiQue answerable dev 全量 2417 条样本上，BM25 单轮检索 all-support-hit@10 仅 6.91%，验证了多轮搜索机制的必要性；在合成多跳集上 Search-Agent 相比 single-shot RAG 显著提升 EM/F1。
```

## 后续开发路线

短期优先级：

```text
1. 跑 MuSiQue 单条 Agent 多跳样例
2. 跑 MuSiQue 小样本 Agent 评测
3. 统计失败案例：检索失败、答案不完整、模型未继续搜索、格式错误
4. 调整 top_k、prompt 和 AnswerJudge
5. 将 MuSiQue 结果写入 README 和 dashboard
```

中期工程增强：

```text
1. ToolRegistry
   统一管理 search、rerank、MCP 等工具。

2. MemoryManager
   将当前 MemoryStore 扩展为 working / episodic / semantic / procedural 多层记忆入口。

3. MCPTool
   将本地检索能力封装为 MCP 工具，支持外部 Agent 或 IDE 调用。

4. API 服务
   使用 FastAPI 暴露 /agent/run、/retrieval/search、/health。

5. Dashboard 增强
   展示检索召回、Agent trace、失败原因和不同 baseline 对比。
```

长期增强：

```text
1. 替换英文 embedding 模型并重跑 MuSiQue 检索召回
2. 增加 reranker 改善 all_support_hit@k
3. 构造训练 trace，用于 SFT 或偏好优化
4. 基于 citation correctness、answer correctness、search cost 设计 reward
5. 对比未训练 Agent 与训练后 Agent 的搜索策略差异
```
