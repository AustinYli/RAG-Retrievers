#!/usr/bin/env bash
set -euo pipefail

python -m rag_bench.run --config configs/scifact_bm25.json
python -m rag_bench.run --config configs/nfcorpus_bm25.json
python -m rag_bench.run --config configs/fiqa_bm25.json
