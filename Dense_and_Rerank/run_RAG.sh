#!/usr/bin/env bash
# Dense retrieval + cross-encoder rerank on TCR-Bench.
# Run from this Dense_and_Rerank/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPUS=("${GPU_ID:-0}")
LLM_API_KEY="${LLM_API_KEY:-xxx}"
LLM_BASE_URL="${LLM_BASE_URL:-xxx}"
TABLE_FORMAT="${TABLE_FORMAT:-markdown}"
CONTENT_TYPE="${CONTENT_TYPE:-question}"

tables_json_path="../data/query_data/TCRAG_Tables.json"
tables_path="../data/table/TCRAG_8k"
questions_json="../data/query_data/TCRAG_Questions.json"

embedding_model_big="Qwen/Qwen3-Embedding-0.6B"
embedding_model_small="Qwen3-Embedding-0.6B"
reranker_model_big="Qwen/Qwen3-Reranker-0.6B"
reranker_model_small="Qwen3-Reranker-0.6B"

gpu_id=${GPUS[0]}

echo "========== Step 1: Dense Retrieval =========="
CUDA_VISIBLE_DEVICES=${gpu_id} python Embedding.py \
  --model-path "${embedding_model_big}" \
  --tables_json_path "${tables_json_path}" \
  --tables_path "${tables_path}" \
  --json-path "${questions_json}" \
  --content_type "${CONTENT_TYPE}" \
  --table_format "${TABLE_FORMAT}" \
  --output-path "./results/${embedding_model_small}/${TABLE_FORMAT}/top5" \
  --key "${LLM_API_KEY}" \
  --url "${LLM_BASE_URL}" \
  --top-k 5

echo "========== Step 2: AAR-CE (Cross-Encoder Rerank) =========="
CUDA_VISIBLE_DEVICES=${gpu_id} python AAR_CE.py \
  --model-path "${reranker_model_big}" \
  --tables_json_path "${tables_json_path}" \
  --tables_path "${tables_path}" \
  --json-path "./results/${embedding_model_small}/${TABLE_FORMAT}/top5/output.json" \
  --content_type "${CONTENT_TYPE}" \
  --table_format "${TABLE_FORMAT}" \
  --output-path "./results_rerank/${reranker_model_small}/${embedding_model_small}/${TABLE_FORMAT}/top5" \
  --key "${LLM_API_KEY}" \
  --url "${LLM_BASE_URL}" \
  --top-k 3

echo "========== Step 3: Compute Metrics =========="
python Compute_Metrics.py \
  --input  "./results/${embedding_model_small}/${TABLE_FORMAT}/top5/output.json" \
  --output "./results/${embedding_model_small}/${TABLE_FORMAT}/top5/full_result.csv" \
  --mode   Embedding

python Compute_Metrics.py \
  --input  "./results_rerank/${reranker_model_small}/${embedding_model_small}/${TABLE_FORMAT}/top5/output.json" \
  --output "./results_rerank/${reranker_model_small}/${embedding_model_small}/${TABLE_FORMAT}/top5/full_result.csv" \
  --mode   Rerank

echo "========== Done =========="
echo "  Dense:  ./results/${embedding_model_small}/${TABLE_FORMAT}/top5"
echo "  Rerank: ./results_rerank/${reranker_model_small}/${embedding_model_small}/${TABLE_FORMAT}/top5"
