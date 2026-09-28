#!/usr/bin/env bash
# Run TableSeek on TCR-Bench and TabFact-test-1k.
# Launch the SGLang server first (see README), then run this script from the code/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8004/v1}"
LLM_API_KEY="${LLM_API_KEY:-xxx}"
NUM_WORKERS="${NUM_WORKERS:-100}"
TABLE_FORMAT="${TABLE_FORMAT:-markdown}"

COMMON_ARGS=(
  --llm-base-url "${LLM_BASE_URL}"
  --llm-api-key "${LLM_API_KEY}"
  --llm-model "${MODEL}"
  --table_format "${TABLE_FORMAT}"
  --content_type question
  --num-workers "${NUM_WORKERS}"
  --n-candidates 5
  --max-select-tables 1
  --max-steps 5
  --preview-chars 800
  --max-matched-rows 5
  --max-rows-per-pattern 1
  --top-k 1
)

echo "[TableSeek] Running TCR-Bench ..."
python TableSeek.py \
  --json-path data/query_data/TCRAG_Questions.json \
  --tables_json_path data/query_data/TCRAG_Tables.json \
  --tables_path data/table/TCRAG_8k \
  --task-type tableqa \
  --output-path results/TCR/${MODEL##*/} \
  "${COMMON_ARGS[@]}"

echo "[TableSeek] Running TabFact-test-1k ..."
python TableSeek.py \
  --json-path data/query_data/tabfact_test_questions_QA_sample1000.json \
  --tables_json_path data/query_data/tabfact_test_tables.json \
  --tables_path data/table \
  --task-type fact_verification \
  --output-path results/TabFact-test-1k/${MODEL##*/} \
  "${COMMON_ARGS[@]}"

echo "[TableSeek] Done."
echo "  TCR-Bench results:       results/TCR/${MODEL##*/}"
echo "  TabFact-test-1k results: results/TabFact-test-1k/${MODEL##*/}"
