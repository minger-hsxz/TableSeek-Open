# Dense Retrieval and Reranking

Dense retrieval and cross-encoder reranking baselines used in this package.
This folder is adapted from the [TCR-Bench](https://github.com/minger-hsxz/TCR-Bench-Open) open-source release and is meant to be run from this directory.

## Layout

```
Dense_and_Rerank/
├── Embedding.py          # Dense retrieval
├── AAR_CE.py             # Cross-encoder rerank over dense top-k (AAR-CE)
├── Compute_Metrics.py    # R@k / GR@k / global DS@1
├── run_RAG.sh            # Example pipeline
├── requirements.txt
└── tools/                # Embedding / rerank helpers and utilities
```

Data are shared from the parent package (`../data/`).

## Setup

```bash
pip install -r requirements.txt
```

## Run

From this directory:

```bash
bash run_RAG.sh
```

The script runs:

1. Dense retrieval (`Embedding.py`) → `./results/...`
2. Cross-encoder rerank (`AAR_CE.py`) → `./results_rerank/...`
3. Metric computation (`Compute_Metrics.py`)

Override via environment variables if needed:

```bash
LLM_API_KEY=xxx LLM_BASE_URL=xxx GPU_ID=0 bash run_RAG.sh
```
