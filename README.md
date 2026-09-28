# TableSeek: Structure-Preserving Agentic Search for Open-Domain Table Retrieval

## Overview

**TableSeek** is a structure-preserving **agentic search** framework for open-domain table retrieval over heterogeneous table corpora.

Open-domain table retrieval seeks tables that contain sufficient evidence for answering a question or verifying a claim. Semantic relevance is often misleading: topically similar tables may lack the required facts, while answer-bearing evidence is typically confined to a few cells whose meaning depends on surrounding schema and table context. Heterogeneous schemas, value formats, and serializations further weaken one-shot matching.

Instead of ranking tables once, an LLM agent iteratively follows sparse clues, inspects schema-preserving previews, identifies schema- and value-level mismatches, and refines its investigation. TableSeek uses cells and schemas as evidence anchors while retaining **complete tables as evidence units**, enabling fine-grained localization without losing the context required for interpretation and answerability checking.

Without relying on retriever training or a precomputed semantic index, TableSeek produces transparent evidence-seeking trajectories and delivers competitive end-to-end performance against strong retrieval-and-reranking pipelines on heterogeneous table benchmarks, including [TCR-Bench](https://github.com/minger-hsxz/TCR-Bench-Open).

---

## Repository Structure

```
TableSeek.py              # End-to-end TableSeek (retrieve + QA)
QA_TableQA.py             # Direct TableQA on gold / retrieved tables
QA_TabFact.py             # Direct TabFact on gold / retrieved tables
run_tableseek.sh          # Example: TableSeek on TCR-Bench + TabFact-test-1k
run_qa.sh                 # Example: direct QA on gold tables
requirements.txt
LICENSE

data/
├── query_data/           # Query / table metadata JSON
└── table/                # Raw table CSVs (e.g., TCRAG_8k)

tableseek/                # Core library
├── patterns.py           # Pattern generation & refine decisions
├── retrieve.py           # Multi-round retrieve + QA loop
├── qa.py                 # Task-specific answering prompts
├── tables.py             # Table corpus loading
├── reformat.py           # Table serialization formats
├── metrics.py            # Evaluation metrics
└── logging_utils.py

Dense_and_Rerank/         # Dense + rerank baselines (adapted from TCR-Bench)
├── Embedding.py          # Dense retrieval
├── AAR_CE.py             # Cross-encoder reranking (AAR-CE)
├── Compute_Metrics.py    # R@k / GR@k / global DS@1
└── run_RAG.sh
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended. An OpenAI-compatible chat API endpoint is required (SGLang recommended below).

### Launch the LLM Server (SGLang)

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --host 0.0.0.0 \
    --port 8004 \
    --tp-size 4 \
    --trust-remote-code \
    --enable-deterministic-inference \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 40960 \
    --max-prefill-tokens 40960 \
    --chunked-prefill-size 4096 \
    --api-key xxx
```

Client defaults match this server: `http://127.0.0.1:8004/v1`, API key `xxx`, model `Qwen/Qwen3-30B-A3B-Thinking-2507`.

---

## Quick Start: TableSeek (Retrieve + QA)

```bash
bash run_tableseek.sh
```

This runs:

1. **TCR-Bench** (`tableqa`) — `data/query_data/TCRAG_Questions.json` over `data/table/TCRAG_8k`
2. **TabFact-test-1k** (`fact_verification`) — TabFact test sample of 1000 claims

Shared search / QA hyperparameters used by the script:

```bash
--max-select-tables 1 \
--max-steps 5 \
--preview-chars 800 \
--max-matched-rows 5 \
--max-rows-per-pattern 1 \
--top-k 1
```

Optional environment overrides:

```bash
LLM_BASE_URL=http://127.0.0.1:8004/v1 LLM_API_KEY=xxx NUM_WORKERS=8 bash run_tableseek.sh
```

### Manual TableSeek example

```bash
python TableSeek.py \
  --json-path data/query_data/TCRAG_Questions.json \
  --tables_json_path data/query_data/TCRAG_Tables.json \
  --tables_path data/table/TCRAG_8k \
  --task-type tableqa \
  --output-path results/TCR/Qwen3-30B-A3B-Thinking-2507 \
  --llm-base-url http://127.0.0.1:8004/v1 \
  --llm-api-key xxx \
  --llm-model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --max-select-tables 1 \
  --max-steps 5 \
  --preview-chars 800 \
  --max-matched-rows 5 \
  --max-rows-per-pattern 1 \
  --top-k 1 \
  --num-workers 8
```

---

## Direct QA (Gold / Retrieved Tables)

These scripts evaluate answering quality when tables are already given (oracle gold tables, or tables from a prior dense / rerank run). They do **not** run TableSeek retrieval.

```bash
bash run_qa.sh
```

This runs gold-table QA on:

1. **TCR-Bench** via `QA_TableQA.py`
2. **TabFact-test-1k** via `QA_TabFact.py`

Optional environment overrides:

```bash
LLM_BASE_URL=http://127.0.0.1:8004/v1 LLM_API_KEY=xxx NUM_WORKERS=8 TABLE_SOURCE=gold bash run_qa.sh
```

`--table-source` options:

| Value | Meaning |
|---|---|
| `gold` | Use `highlighted_table` from the query JSON |
| `dense` | Use top-k keys from `output_text` (prior retrieval dump) |
| `rerank` | Use top-k keys from `output_text_rerank` |

---

## Dense and Rerank Baselines

The `Dense_and_Rerank/` folder contains dense retrieval and cross-encoder reranking baselines, adapted from the [TCR-Bench](https://github.com/minger-hsxz/TCR-Bench-Open) open-source release. Run from `Dense_and_Rerank/`:

```bash
cd Dense_and_Rerank
bash run_RAG.sh
```

See `Dense_and_Rerank/README.md` for details. It reads shared data from `../data/`.

---

## Task Types (TableSeek)

| `--task-type` | Description |
|---|---|
| `tableqa` | Short-answer table QA (TCR-Bench) |
| `fact_verification` | Claim verification (TabFact) |

---

## Outputs

Under `--output-path`:

- `output.json` / `output.jsonl` — predictions (and retrieval traces for TableSeek)
- `patterns.json` / `patterns.jsonl` — TableSeek search patterns and step logs
- `result.csv` — aggregate metrics
- `action_stats.json` / `action_stats.csv` — TableSeek agent action statistics
- `log.txt` — run log

---

## Data Notes

- **TCR-Bench** query/table metadata and CSV tables are included under `data/` for reproducing TableSeek experiments. For the full diagnostic benchmark release, see [TCR-Bench-Open](https://github.com/minger-hsxz/TCR-Bench-Open).
- **TabFact** evaluation uses inline `table_data` in the provided JSON files (no separate CSV dump required).
- Please respect the original licenses of any third-party datasets you redistribute or extend.

---

## License

- Code: [MIT](LICENSE)
- Bundled evaluation data inherits the licenses of the respective source datasets (e.g., TCR-Bench / TabFact). See those projects for details.

---

## Notes

- All paths in this package are **relative** (TableSeek / QA from the repo root; dense/rerank from `Dense_and_Rerank/`).
- No dense retriever training or precomputed semantic index is required for TableSeek.
- TableSeek keeps full tables as evidence units while using schema/cell-level clues for iterative localization.
