#!/usr/bin/env bash
set -euo pipefail

python -m rag_bench.run --config configs/scifact_bm25_smoke.json
python -m rag_bench.run --config configs/scifact_dense_minilm_smoke.json
python -m rag_bench.run --config configs/scifact_hybrid_minilm_smoke.json
