# RAG Benchmark Harness

This repo implements repeatable retrieval and RAG evaluation. Track A benchmarks BM25, exact dense retrieval, hybrid RRF, and cross-encoder reranking on BEIR. Track B adds evidence retrieval, local answer generation, faithfulness, and abstention evaluation on MultiHop-RAG.

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

## Track B: MultiHop-RAG

Download and audit the public files directly from their authoritative Hugging Face repository. The data stays under ignored `data/`; the computed audit is tracked:

```bash
python scripts/inspect_multihop_rag.py
```

Run Stage 1 in dependency order. The runner checkpoints the expensive reranker under `cache/` and resumes it after interruption:

```bash
python scripts/run_track_b_stage1.py --stage bm25
python scripts/run_track_b_stage1.py --stage dense
python scripts/run_track_b_stage1.py --stage hybrid
python scripts/run_track_b_stage1.py --stage rerank --rerank-batch-size 16
python scripts/compare_track_b_stage1.py --samples 10000
python scripts/check_track_b_evidence_metrics.py
python scripts/analyze_track_b_retrieval.py
```

The corpus contains full news articles, so Track B uses deterministic 256-word chunks with 64-word overlap and indexes `title`, `source`, and `published_at` alongside every chunk. It reports both source-document recall and stricter evidence-fact recall; `context_sufficiency@k` is one only when every annotated fact is present in the selected chunks. This is complete gold-evidence coverage, not an unconditional accuracy ceiling, because annotations may not enumerate every usable support path.

Install the pinned local generator and run the oracle-context gate before any retrieved-context generation:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
python scripts/run_track_b_generation.py --mode oracle --max-queries 20
```

Once the full oracle gate passes, run three repeats on the frozen 20-question
development slice for the determinism check, then vary only the retrieval run:

```bash
python scripts/run_track_b_generation.py --mode oracle --max-queries 20 --repeat 1
python scripts/run_track_b_generation.py --mode oracle --max-queries 20 --repeat 2
python scripts/run_track_b_generation.py --mode oracle --max-queries 20 --repeat 3

python scripts/analyze_track_b_determinism.py \
  results/track_b_smoke_generation_artifacts/<repeat-1>.jsonl \
  results/track_b_smoke_generation_artifacts/<repeat-2>.jsonl \
  results/track_b_smoke_generation_artifacts/<repeat-3>.jsonl \
  --output results/track_b_determinism.json

python scripts/run_track_b_generation.py \
  --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64
```

The first 20 answerable questions are a frozen prompt-development slice because their raw outputs were inspected while fixing the response contract. Full generation runs exclude those IDs from their artifacts and record both the ID-set hash and evaluation-partition version; headline generation metrics therefore use 2,235 held-out answerable questions.

The completed held-out oracle gate is 0.7955 EM and 0.8034 token F1. Three temperature-zero repeats on the frozen development slice produced identical predictions (0.000 EM/F1 range), so no run averaging is required by the measured determinism gate. Oracle p50/p95 latency is 1.812/2.606 seconds per sequential local request. The summary flags and excludes 45 impossible Ollama component-duration ledgers while retaining every well-formed total request duration.

After generating with BM25, dense, and dense plus reranker, compare the exact
generation run names as one Holm-corrected family:

```bash
python scripts/compare_track_b_generation.py \
  --ordered-run-names <bm25-generation-run>,<dense-generation-run>,<reranker-generation-run>

python scripts/analyze_track_b_generation.py \
  --generation-artifact results/track_b_generation_artifacts/<run>.jsonl \
  --output results/track_b_failure_attribution.json
```

Position sensitivity uses three runs over the same retriever and fixed chunk set,
followed by one six-test Holm family:

```bash
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name <retriever> --evidence-position first \
  --evaluation-sample-size 300
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name <retriever> --evidence-position middle \
  --evaluation-sample-size 300
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name <retriever> --evidence-position last \
  --evaluation-sample-size 300
python scripts/compare_track_b_positions.py \
  --ordered-run-names <first-run>,<middle-run>,<last-run>
```

The abstention intervention requires matched retrieved runs with the instruction
off and on. Its analyzer verifies that the saved prompt templates differ only by
the frozen instruction, then jointly corrects the null-abstention and answerable
false-abstention tests. Use the same frozen 300-query hash sample for both runs:

```bash
python scripts/compare_track_b_abstention.py \
  --off-run-name <instruction-off-run> --on-run-name <instruction-on-run>
```

`--evaluation-sample-size` is reserved for secondary experiments, uses a
versioned SHA-256 sampler, and records the seed and selected-ID hash. The full E1
retriever comparison does not use it.

Generation artifacts include the exact rendered context and its hash, answer, prompt hash, context-builder and response-parser versions, upstream retrieval artifact hash, model tag and digest, quantization, runtime, seed, temperature, context window, 96-token output cap, EM/F1, and abstention decision. The NLI scorer emits a threshold sensitivity curve and accepts a hand-labeled calibration CSV:

```bash
python scripts/score_track_b_faithfulness.py \
  --generation-artifact results/track_b_generation_artifacts/<run>.jsonl \
  --summary-output results/track_b_faithfulness_summary.json

python scripts/sample_faithfulness_calibration.py \
  --claim-artifact results/track_b_generation_artifacts/<run>.claims.jsonl
```

The calibration CSV deliberately omits the NLI score and includes the question,
claim, and full context needed to assign each `human_supported` label. Do not
quote a headline faithfulness rate until those 50 labels and Cohen's kappa are
present.

Retrieved generation defaults to five 256-word chunks and a strict 1,280-word rendered-context budget that includes block labels and retrieval metadata. Body words are allocated across the fixed chunk set independently of display order. For chunking experiments, raise `--retrieved-top-k` while keeping `--max-context-words 1280`; for position sensitivity, repeat the same run with `--evidence-position first`, `middle`, and `last`.

E5 supports fixed, paragraph-structural, and BGE-based semantic segmentation. Non-fixed strategies require zero overlap; semantic boundaries are persisted and hash-verified. Run each retrieval configuration first, then use rank-order greedy packing so all four variants consume the same total rendered-context budget:

```bash
python scripts/run_track_b_stage1.py --stage bm25 --overlap-words 0
python scripts/run_track_b_stage1.py --stage bm25 \
  --chunking-strategy structural --overlap-words 0
python scripts/run_track_b_stage1.py --stage bm25 \
  --chunking-strategy semantic --overlap-words 0

python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name <chunking-run> --retrieved-top-k 100 \
  --context-packing greedy --max-context-words 1280 \
  --evaluation-sample-size 300

python scripts/compare_track_b_chunking.py \
  --baseline-run-name <fixed-overlap-generation-run> \
  --candidate-run-names <no-overlap-run>,<structural-run>,<semantic-run>
```

See [TRACK_B.md](TRACK_B.md) for metric definitions, gates, and current findings.
