#!/usr/bin/env bash
# Run direct QA on TCR-Bench (TableQA) and TabFact-test-1k.
# Launch the SGLang server first (see README), then run this script from the code/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8004/v1}"
LLM_API_KEY="${LLM_API_KEY:-xxx}"
NUM_WORKERS="${NUM_WORKERS:-100}"
TABLE_FORMAT="${TABLE_FORMAT:-markdown}"
TABLE_SOURCE="${TABLE_SOURCE:-gold}"
TOP_K="${TOP_K:-1}"

COMMON_ARGS=(
  --url "${LLM_BASE_URL}"
  --key "${LLM_API_KEY}"
  --model-path "${MODEL}"
  --table-format "${TABLE_FORMAT}"
  --table-source "${TABLE_SOURCE}"
  --top-k "${TOP_K}"
  --num-workers "${NUM_WORKERS}"
)

echo "[QA] Running TableQA on TCR-Bench ..."
python QA_TableQA.py \
  --json-path data/query_data/TCRAG_Questions.json \
  --table-json-path data/query_data/TCRAG_Tables.json \
  --tables-path data/table/TCRAG_8k \
  --output-path results/QA_TableQA/${MODEL##*/} \
  "${COMMON_ARGS[@]}"

echo "[QA] Running TabFact-test-1k ..."
python QA_TabFact.py \
  --json-path data/query_data/tabfact_test_questions_QA_sample1000.json \
  --table-json-path data/query_data/tabfact_test_tables.json \
  --output-path results/QA_TabFact/${MODEL##*/} \
  "${COMMON_ARGS[@]}"

echo "[QA] Done."
echo "  TableQA results:  results/QA_TableQA/${MODEL##*/}"
echo "  TabFact results:  results/QA_TabFact/${MODEL##*/}"
