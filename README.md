# RAG Benchmark Harness

This repo implements Track A from the project brief: repeatable BEIR retrieval benchmarks for BM25, exact dense retrieval, hybrid RRF, and optional cross-encoder reranking.

The fixed grid is in `results/track_a_runs.csv`. See [RESULTS.md](RESULTS.md) for the findings, statistical qualifications, and decision log.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run a tiny smoke test first. Smoke runs write to `results/smoke_runs.csv`, not the real benchmark table:

```bash
python -m rag_bench.run --config configs/scifact_bm25_smoke.json
# or:
scripts/run_smoke.sh
```

Then run the intended Track A baselines:

```bash
python -m rag_bench.run --config configs/scifact_bm25.json
python -m rag_bench.run --config configs/nfcorpus_bm25.json
python -m rag_bench.run --config configs/fiqa_bm25.json
# or:
scripts/run_track_a_bm25.sh
```

Run the SciFact BM25 analyzer/parameter gate:

```bash
python scripts/run_bm25_gate.py
```

Dense and hybrid runs:

```bash
python -m rag_bench.run --config configs/scifact_dense_minilm.json
python -m rag_bench.run --config configs/scifact_hybrid_minilm.json
python -m rag_bench.run --config configs/scifact_rerank_minilm.json
python -m rag_bench.run --config configs/scifact_dense_minilm_rerank.json
python -m rag_bench.run --config configs/scifact_hybrid_minilm_rerank.json
```

Run the complete fixed grid and post-hoc sweeps:

```bash
python scripts/run_track_a_grid.py
python scripts/run_fusion_sweeps.py --force
python scripts/compare_track_a.py --samples 10000
python scripts/compare_rrf_k_sweep.py --samples 10000
python scripts/compare_fusion_winners.py --samples 10000
```

Reranker sweeps persist both ranked lists and per-query metrics, record the actual Torch device, and resume by skipping completed run names. Their `rerank_latency_p50_ms` and `rerank_latency_p95_ms` fields are directly measured; total query percentiles are labeled `component_percentile_sum` because they add the persisted base and reranker component percentiles:

```bash
python scripts/run_rerank_sweep.py \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --trust-remote-code \
  --datasets fiqa \
  --base-run-names fiqa_dense_bge_512 \
  --rerank-top-k-values 10,20,50,100

python scripts/compare_rerank_sweeps.py \
  --rerank results/rerank_bge_probe.csv \
  --output results/rerank_bge_probe_comparisons.csv \
  --holm-output results/rerank_bge_probe_comparisons_holm.csv \
  --include-adjacent-depths
```

## Notes

- BEIR corpora are already passage-like documents, so this track intentionally does not benchmark chunking.
- Embeddings and BM25 indices are cached under `cache/` so repeated runs are much faster.
- Dense retrieval defaults to exact NumPy inner-product search over L2-normalized embeddings. Set `retriever.index_backend` to `faiss` to use FAISS `IndexFlatIP`.
- E5 and BGE model prefixes are applied automatically when their model names are detected.
- Reranking is optional because it downloads a cross-encoder and is much more expensive than first-stage retrieval.
- No model is trained or fine-tuned here. The included sweeps use BEIR test qrels, so selected optima are exploratory and should be validated on a held-out dataset or split before deployment.

## Useful Config Fields

- `dataset`: BEIR dataset name, for example `scifact`, `nfcorpus`, or `fiqa`
- `split`: usually `test`
- `max_corpus_docs` / `max_queries`: optional smoke-test limits
- `retriever.type`: `bm25`, `dense`, `hybrid`, or `rerank`
- `retriever.dense_model`: Sentence Transformers model name
- `retriever.index_backend`: `numpy` or `faiss`
- `retriever.analyzer`: `word-lower-v1`, `word-lower-stop-v1`, `word-lower-bigram-v1`, or `word-lower-stop-porter-v1`
- `retriever.k1` / `retriever.b` / `retriever.epsilon`: BM25 parameters
- `retriever.reranker_model`: CrossEncoder model name
- `retriever.max_seq_length`: explicit truncation length for dense and reranker models
- `retriever.normalize_embeddings`: makes exact inner product equivalent to cosine similarity
- `metrics.cutoffs`: list of metric cutoffs, defaults to `[1, 5, 10, 100]`

## Output

Each real run appends a row to `results/runs.csv` with config metadata, timing, artifact paths, and metrics such as:

- `ndcg@10`
- `recall@10`
- `mrr@10`
- `map@10`

`map@k` follows the `pytrec_eval` `map_cut.k` convention: precision at relevant ranks up to `k`, divided by the full number of relevant documents for the query.

The row also includes timing fields:

- `load_seconds`: dataset loading and any first-time download/unzip
- `index_seconds`: BM25 indexing or dense corpus embedding/cache loading
- `search_seconds`: query-time retrieval/reranking work
- `throughput_ms_per_query`: `search_seconds / num_queries`
- `query_latency_p50_ms` and `query_latency_p95_ms`: query-by-query latency when measured by the retriever

Full ranked outputs are written as JSONL under `results/artifacts/`, and per-query metric CSVs are written beside them.

If metric code changes, recompute metrics from persisted rankings without rerunning retrieval:

```bash
python scripts/rescore_results.py results/track_a_runs.csv
```

## Comparing Runs

Use persisted per-query metrics for paired bootstrap confidence intervals:

```bash
python -m rag_bench.compare \
  --baseline results/artifacts/<baseline>.per_query.csv \
  --candidate results/artifacts/<candidate>.per_query.csv \
  --metric ndcg@10 \
  --output results/comparisons.csv
```

The test suite cross-checks nDCG, recall, and MAP against `pytrec_eval`, including a query with more relevant documents than the cutoff so the MAP denominator convention is exercised.

To repeat that check over every persisted Track A ranking:

```bash
python scripts/check_metrics_pytrec_eval.py
```
