# Track B: Generation and Grounding

Track B asks whether retrieval improvements change answers, and separates two failure modes: missing evidence and failure to use evidence that was retrieved. No model is trained or fine-tuned.

## Dataset Audit

The audit is computed from the downloaded JSON files by `scripts/inspect_multihop_rag.py`, not copied from the paper. The inputs come from the authoritative [MultiHop-RAG dataset](https://huggingface.co/datasets/yixuantt/MultiHopRAG); the [project repository](https://github.com/yixuantt/MultiHop-RAG) and [paper](https://arxiv.org/abs/2401.15391) are used only for provenance and cross-checks.

| Item | Verified value |
| --- | ---: |
| Questions | 2,556 |
| Answerable questions | 2,255 |
| Null questions | 301 |
| Comparison / inference / temporal | 856 / 816 / 583 |
| Corpus documents | 609 full news articles |
| Mean article length | 1,746 whitespace-delimited words |
| Evidence annotations | 6,084 |
| Mean evidence annotations per answerable question | 2.698 |
| Mean unique evidence documents per answerable question | 2.620 |
| Questions repeating an evidence document | 168 |

Every evidence URL maps to one corpus article, and every annotated fact occurs verbatim in that article. The source files are pinned by SHA-256 in `results/track_b_dataset_audit.json`.

The released files contain 1,079 questions with two evidence annotations and 778 with three. Table 4 of the paper reports 1,078 and 779 respectively, a one-record swap that leaves the total unchanged. Results in this repository follow the released files.

## Retrieval Unit

Unlike BEIR, this corpus is made of full articles. The initial fixed representation is:

- 256 whitespace-delimited words per chunk
- 64-word overlap
- 5,636 chunks, or 9.255 chunks per article on average
- indexed metadata: title, publisher, and publication timestamp

Whitespace words are deliberately named as the unit. They are not claimed to reproduce the paper's 256-token LlamaIndex splitter. Later chunking comparisons must hold the retrieved context budget fixed.

## Evidence Metrics

`evidence_recall@k` is the fraction of annotated evidence facts that occur in the top-k retrieved chunks. `evidence_doc_recall@k` is the fraction of unique source articles represented in those chunks. These differ whenever the right article is retrieved but the selected chunk does not contain the annotated fact.

`context_sufficiency@k` is binary per question: one only when every annotated fact occurs in the top-k chunks. Its mean measures complete gold-evidence coverage. It is a strong attribution diagnostic, but not a mathematical accuracy ceiling: annotations may be incomplete or redundant, and a model can sometimes answer from partial evidence or memorized knowledge despite the prompt constraint.

Null questions are excluded from evidence metrics and retained for the later abstention experiment.

## Stage 1 Gate

### Preregistered out-of-sample fusion prediction

Recorded before running MultiHop-RAG hybrid retrieval: the frozen test uses the same equal-weight reciprocal-rank fusion as all nine Track A gap-law cells (`rrf_k=60`, BM25 weight 1.0, BGE weight 1.0), applied to the already persisted BM25 and BGE-base top-100 rankings. No MultiHop-RAG fusion parameter is tuned.

The directional, rank-ordered prediction is that **hybrid minus BGE dense on MultiHop-RAG evidence recall@5 will be larger than every hybrid-minus-dense nDCG@10 gain in the nine-cell Track A fit**. This extrapolates from MultiHop-RAG's dense-minus-BM25 evidence-recall gap of -0.1645, which is more negative than any branch gap in the fit. It does not predict a numerical delta across unlike metrics, and it does not predict that hybrid will beat BM25 outright. Context sufficiency and comparisons against BM25 are secondary outcomes; the preregistered claim is the hybrid lift over dense.

Status at preregistration: not run. The result must be appended without editing the prediction above.

The three completed rows use the frozen Track A BM25 configuration, `BAAI/bge-base-en-v1.5` at immutable revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, and `BAAI/bge-reranker-v2-m3` at immutable revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Dense search is exact normalized inner product, and the reranker reorders its top 20 while retaining the full top-100 tail. At `k=5`:

| Retriever | Evidence recall@5 | Context sufficiency@5 |
| --- | ---: | ---: |
| BM25 | 0.4707 | 0.1831 |
| BGE-base dense | 0.3062 | 0.0625 |
| BGE-base dense + BGE reranker@20 | 0.4345 | 0.1494 |

The reranker recovers `+0.1283` evidence recall@5 over dense, with a paired 10,000-sample 95% bootstrap interval of `[+0.1172, +0.1396]`. It also recovers `+0.0869` context sufficiency@5, with `[+0.0736, +0.1007]`. Both survive Holm correction across the six planned Stage 1 tests.

It does not overtake BM25. Reranked dense remains `-0.0362` behind BM25 on evidence recall@5 (`[-0.0489, -0.0237]`) and `-0.0337` behind on context sufficiency@5 (`[-0.0497, -0.0182]`). The planned expectation that dense must beat sparse therefore does not hold for either the base dense model or its reranked top 20 on this benchmark.

This is not attributed to a loader error: URL and fact mappings are complete, metadata is indexed, the [BGE query instruction](https://huggingface.co/BAAI/bge-base-en-v1.5) is present, embeddings are normalized, search is exact, and an independent implementation reproduces official-style Hits@10 for all three artifacts. The dataset strongly favors lexical source routing: 2,242 of 2,255 answerable questions name at least one gold publisher, and 2,091 name every gold publisher. BM25's advantage also appears independently in comparison, inference, and temporal questions. MultiHop-RAG's paper tested `bge-large-en-v1.5`, not the frozen `bge-base-en-v1.5`, and did not report a BM25 comparison. The brief's ordering assumption is therefore treated as falsified for this frozen grid, with the discrepancy disclosed rather than relabeled as a software defect.

On this Apple Silicon machine, measured p50 search latency was 14.6 ms/query for BM25 and 8.8 ms/query for dense. Reranking was 2,263 ms/query using cross-query batch-amortized timing, so it is not a serving-latency measurement and is not directly interchangeable with the sequential base timings. The complete rows, per-query artifacts, comparisons, and Holm-adjusted results are persisted under `results/track_b_stage1*`.

## Generation Gates

1. Oracle context: run answerable questions with annotated facts directly. If EM/F1 is poor, change the fixed generator before retrieval comparisons.
2. Determinism: repeat one frozen configuration three times. Report the spread before interpreting retriever deltas.
3. Retrieved context: vary only BM25, BGE-base dense, and BGE-base plus reranker. Use the same top-5 chunks, prompt, generator, temperature, seed, and context window.

Correctness uses frozen SQuAD-style exact match and token F1 over a concise `Answer:` field. The same frozen response contract requests a complete `Claim:` that explicitly states the answer, because an entity fragment alone is not a valid NLI hypothesis; malformed responses are preserved and flagged. Abstention always reports both null-query abstention and answerable false abstention. Faithfulness uses the [DeBERTa-v3-base NLI cross-encoder](https://huggingface.co/cross-encoder/nli-deberta-v3-base) with a threshold curve; each claim is scored against each context block and receives the maximum entailment probability, avoiding silent truncation of later chunks. A headline unsupported-claim rate requires agreement against 50 hand labels, reported with Cohen's kappa.

Retrieved prompts use a strict 1,280-word rendered-context budget, including block labels, titles, publishers, and timestamps. The body budget is balanced across the fixed top-five chunk IDs using a chunk-ID-stable allocation, so evidence-position experiments change order without also truncating the final position more aggressively. This is a whitespace-word budget, not a claim of exact Qwen-token matching.

The paper's released [`qa_evaluate.py`](https://github.com/yixuantt/MultiHop-RAG/blob/main/qa_evaluate.py) counts a prediction as correct when it shares any whitespace token with the gold answer. Its published "accuracy" values are therefore not directly comparable with this harness's stricter EM; they are contextual references only.

The final frozen prompt passed a 20-question oracle smoke test at 0.850 EM, 0.883 token F1, 1.000 response-format validity, and 0.050 false abstention. No response reached the 96-token output cap, and the longest claim contained 20 words under the 25-word contract. Because those questions were inspected while repairing the response contract, their IDs are frozen and hashed as a prompt-development slice. They are physically excluded from every full oracle and retrieved generation run, leaving 2,235 answerable evaluation questions and all 301 null questions. The smoke result selects the runnable prompt contract; it is not reported as the final oracle estimate.

The definitive held-out oracle gate reached **0.7955 EM and 0.8034 token F1** on all 2,235 evaluation questions. Performance varies materially by question type: inference is strongest at 0.9156 EM, comparison reaches 0.7976, and temporal questions are the bottleneck at 0.6252. Response-format validity is 0.9808, answerable false abstention is 0.0125, and no response reached the 96-token cap. Sequential local generation took 1.812 s/query at p50 and 2.606 s/query at p95 with short oracle evidence.

Ollama returned impossible component-duration fields on 45 of 2,235 responses even though each total request duration remained well formed. Raw values remain in the artifact; component sums exclude those rows and the summary records a 0.9799 component-ledger validity rate. Total request latency uses all rows.

The temperature-zero determinism gate produced identical predictions in three repeats over the frozen 20-question development slice: EM was 0.850 in every run, token F1 was 0.883 in every run, and exact prediction agreement was 1.000. The observed run-level noise floor is therefore zero on this probe, not a guarantee that every platform or larger sample is deterministic.

## Chunking Experiment

E5 exposes three segmentation strategies behind total cache and run keys: fixed whitespace-word windows, paragraph-aware structural packing, and adjacent-sentence semantic breaks from the pinned BGE-base encoder. Semantic boundaries are persisted as a hash-verified chunk artifact and are never silently recomputed during generation.

Chunking comparisons use greedy rank-order packing under the same 1,280-word rendered-context budget, including labels and metadata. This is intentionally separate from the balanced top-five packer used by E1 and the evidence-position experiment. The planned E5 family holds BM25 and every generation control fixed and compares fixed windows without overlap, the current 64-word overlap baseline, structural chunks, and semantic chunks on the same versioned 300-query evaluation sample.
