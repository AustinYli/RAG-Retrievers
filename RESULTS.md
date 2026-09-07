# Results and Decision Log

## Evaluation Boundary

No model was trained or fine-tuned. All dense retrievers and cross-encoders are pretrained models used as-is.

The benchmark evaluates on the BEIR test queries and qrels. The fixed Track A grid is a direct benchmark, while the `rrf_k`, branch-weight, candidate-depth, and reranker-depth argmaxes are post-hoc analyses on those same test queries. Their curves and failure modes are useful evidence, but their selected maxima are not unbiased estimates of performance on unseen data.

## Harness Validation

The exact BGE dense path nearly reproduces the official `BAAI/bge-base-en-v1.5` model-card results:

| Dataset | This harness nDCG@10 | Model card nDCG@10 | Difference |
| --- | ---: | ---: | ---: |
| SciFact | 0.74039 | 0.74039 | +0.00000 |
| NFCorpus | 0.37438 | 0.37389 | +0.00049 |
| FiQA | 0.40623 | 0.40646 | -0.00023 |

Source: [BAAI/bge-base-en-v1.5 model card](https://huggingface.co/BAAI/bge-base-en-v1.5). This is an independent reproduction on a dense retriever family, complementing the BM25 reference check.

nDCG, recall, and MAP are cross-checked against `pytrec_eval`. The MAP test deliberately includes 15 relevant documents at cutoff 10, and the nDCG fixture includes graded relevance. This caught and fixed the prior `min(relevant, cutoff)` MAP denominator, which inflated NFCorpus MAP. A second check covers 252 metric/run combinations over all 21 persisted Track A rankings; the maximum absolute difference is `1.11e-16`. Strict surrogate scores preserve the harness's deterministic document-ID tie break while isolating the metric formulas from `pytrec_eval`'s different raw-score tie policy. All persisted Track A and fusion metrics were rescored; nDCG and recall did not change.

## Fixed Track A Grid

The fixed 21-run grid uses the same frozen Porter BM25 configuration, exact normalized dense search, `top_k=1000`, and `max_seq_length=512` across all three datasets. The best row in each retrieval family is:

| Dataset | BM25 | Best dense | Best equal-weight hybrid |
| --- | ---: | ---: | ---: |
| SciFact | 0.68417 | 0.74039 (BGE) | 0.74590 (E5) |
| NFCorpus | 0.32692 | 0.37438 (BGE) | 0.37520 (BGE) |
| FiQA | 0.24399 | 0.40623 (BGE) | 0.37621 (E5) |

Values are nDCG@10. This fixed grid shows that fusion is conditional: it is slightly best on SciFact and NFCorpus, while every equal-weight FiQA hybrid trails its corresponding dense branch. The full table, including recall and measured p50/p95 latency, is `results/track_a_runs.csv`.

## Finding 1: Branch Imbalance Predicts Equal-Weight Fusion Failure

Across the nine dataset/model cells, the gap between dense and BM25 quality is strongly negatively associated with the benefit of equal-weight RRF over dense alone:

- Spearman rho: -0.983
- Pearson r: -0.938
- OLS slope: -0.456

When the branches are close, equal-weight fusion tends to help. When dense is far stronger, equal weighting gives too much influence to the weaker sparse branch and can hurt. This is a post-hoc association over nine cells, not yet a validated universal threshold.

The result explains the grid coherently: MiniLM receives the largest hybrid gains where it trails BM25, while all three FiQA dense models beat BM25 by at least 0.117 and all three lose quality under equal-weight `k=60` fusion. The auditable observations are in `results/gap_law.csv`.

## Finding 2: Conventional RRF k Was Not Robust

Sweeping `rrf_k` while holding the BGE dense and BM25 rankings fixed improved the equal-weight result on SciFact and FiQA:

| Dataset | k=60 | Post-hoc best k | Best nDCG@10 | Delta | 95% paired CI | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SciFact | 0.73599 | 2 | 0.74868 | +0.01269 | [+0.00001, +0.02618] | 0.0996 |
| NFCorpus | 0.37520 | 10 | 0.37663 | +0.00143 | [-0.00325, +0.00615] | 0.5308 |
| FiQA | 0.36392 | 2 | 0.38214 | +0.01821 | [+0.00665, +0.02985] | 0.0048 |

FiQA survives Holm correction; SciFact is borderline before correction and does not survive the three-test family. The fixed grid remains at `k=60` for provenance, while `results/rrf_k_comparisons_holm.csv` records the post-hoc tuning result.

## Finding 3: Weighted Fusion Has One Substantive Win

Compared with the best dense baseline, the selected weighted-fusion rows are:

| Dataset | Dense weight | Delta nDCG@10 | 95% paired CI | Holm p | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| SciFact | 4 | +0.01710 | [+0.00468, +0.02999] | 0.0210 | Interior turnover and practically useful gain |
| NFCorpus | 20 | +0.00435 | [+0.00081, +0.00817] | 0.0296 | Statistically detectable, practically small plateau |
| FiQA | 50 | +0.00234 | [+0.00034, +0.00464] | 0.0296 | Almost-dense plateau; use dense alone |

The confidence intervals compare the selected rows with dense but do not correct for choosing the maximum over the same test queries. The substantive conclusion is SciFact's interior optimum. On FiQA, increasing the dense weight mostly turns off a harmful sparse branch.

The NFCorpus asymmetric-depth control is effectively flat. Full BM25/dense depth `1000/1000` scores `0.37520`; the other `{100, 1000} x {100, 1000}` settings range from `0.37396` to `0.37456`. A requested BM25 depth of 100 realizes only 70.7 candidates per query on average because documents with nonpositive sparse scores are discarded. The recorded candidate depth is therefore an upper bound, while the realized-depth columns describe what was actually fused.

## Finding 4: The Reranker Model Was a Confound

The first sweep used `cross-encoder/ms-marco-MiniLM-L-6-v2`. After regenerating all 30 rows with corrected MAP and persistent artifacts, the best tested depth for each base is:

| Dataset | Base | Best depth | Delta nDCG@10 |
| --- | --- | ---: | ---: |
| SciFact | E5 hybrid | 10 | -0.02418 |
| SciFact | BGE dense | 10 | -0.01741 |
| NFCorpus | BGE hybrid | 10 | -0.00121 |
| NFCorpus | BGE dense | 10 | +0.00083 |
| FiQA | BGE dense | 10 | -0.00774 |
| FiQA | E5 dense | 20 | +0.00754 |

No best-depth MiniLM improvement survives the 54-comparison Holm family, and quality usually falls as depth increases. Several deep negative deltas are significant. This supports a narrow conclusion about this MiniLM cross-encoder, not reranking as a method.

Replacing it with `BAAI/bge-reranker-v2-m3` produced a domain-dependent result over the BGE dense base:

| Dataset | Depth | Base nDCG@10 | Reranked nDCG@10 | Delta | 95% paired CI | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FiQA | 20 | 0.40623 | 0.43521 | +0.02898 | [+0.01397, +0.04444] | 0.0026 |
| SciFact | 20 | 0.74039 | 0.74986 | +0.00947 | [-0.01413, +0.03308] | 1.0000 |
| NFCorpus | 20 | 0.37438 | 0.36233 | -0.01204 | [-0.02722, +0.00289] | 0.9280 |

The implementation matches the official reranker recipe: raw query/passage pairs, classification logits where larger means more relevant, and truncation at 512 tokens. The model card's BEIR evaluation reranks the top 100 from `bge-en-v1.5-large`, whereas this control starts from `bge-base-en-v1.5` and sweeps multiple depths. These are therefore independent domain controls, not a reproduction of the card's reranking scores. Source: [BAAI/bge-reranker-v2-m3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3).

All four FiQA depths improve significantly over the base after Holm correction. The numerical maximum is top-50 at `0.43647`, but top-20 is the practical knee: moving from 20 to 50 adds only `+0.00126` nDCG@10 (95% CI `[-0.00738, +0.01030]`) while directly measured p50 reranking latency rises from 2.11 to 4.54 seconds per query. SciFact's positive point estimate and NFCorpus's negative point estimate are both inconclusive. The control therefore rejects both blanket claims: reranking is neither generally useless nor generally beneficial, and reranker strength alone does not determine the outcome.

Recall@100 is unchanged at every depth within each dataset, validating that the implementation reranks the head and preserves the unscored tail. The confidence family includes each base-to-reranker comparison and each adjacent-depth comparison across all three datasets.

The original BGE reranker CSV omitted the device. Sentence Transformers auto-selected `mps:0` on this host, and the completed rows now record it explicitly. At FiQA top-20, directly measured reranker p50/p95 are 2.11/2.85 seconds. The `query_latency_p50_ms` and `query_latency_p95_ms` sweep columns are labeled `component_percentile_sum`: they estimate total latency by adding base and reranker percentiles, rather than measuring the percentile of paired end-to-end samples. These local Apple MPS measurements are not claims about T4, A10, or production GPU serving latency.

## Decision Log

1. The initial BM25 result was below its reference, so analyzer and parameter sweeps were run before interpreting hybrid gains. Porter stemming raised SciFact BM25 and reduced the apparent hybrid delta.
2. Equal-weight RRF looked inconsistent across datasets. Comparing all nine branch gaps exposed a single branch-imbalance mechanism rather than three unrelated outcomes.
3. Weighted fusion recovered SciFact, but FiQA and NFCorpus formed nearly flat, practically negligible plateaus. Those two are not resume headline gains.
4. MiniLM suggested a negative reranking result. Recognizing model capacity and domain as confounds led to a stronger BGE control, which improved FiQA significantly but not SciFact or NFCorpus. The correct story is that reranker choice, domain, and depth interact, and quality must be weighed against measured hardware-specific latency.
