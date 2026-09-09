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

**Result appended after execution:** confirmed. Frozen fusion raises evidence recall@5 from 0.3062 to 0.4060, a +0.0998 paired lift with a 10,000-sample 95% bootstrap interval of [+0.0900, +0.1098]. That exceeds the largest hybrid-minus-dense gain among the nine Track A cells (0.0685), while respecting the preregistered caveat that the Track A values use nDCG@10. Context sufficiency@5 rises from 0.0625 to 0.1259 (+0.0634 [+0.0519, +0.0749]). Both survive Holm correction in the four-test prediction family. Hybrid remains below BM25 by 0.0647 evidence recall and 0.0572 context sufficiency, also significant.

The four completed rows use the frozen Track A BM25 configuration, `BAAI/bge-base-en-v1.5` at immutable revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, and `BAAI/bge-reranker-v2-m3` at immutable revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Dense search is exact normalized inner product, frozen hybrid uses equal-weight RRF at `k=60`, and the reranker reorders the dense top 20 while retaining the full top-100 tail. At `k=5`:

| Retriever | Evidence recall@5 | Context sufficiency@5 |
| --- | ---: | ---: |
| BM25 | 0.4707 | 0.1831 |
| BGE-base dense | 0.3062 | 0.0625 |
| Frozen BM25 + BGE RRF | 0.4060 | 0.1259 |
| BGE-base dense + BGE reranker@20 | 0.4345 | 0.1494 |

The reranker recovers `+0.1283` evidence recall@5 over dense, with a paired 10,000-sample 95% bootstrap interval of `[+0.1172, +0.1396]`. It also recovers `+0.0869` context sufficiency@5, with `[+0.0736, +0.1007]`. Both survive Holm correction across the six planned Stage 1 tests.

It does not overtake BM25. Reranked dense remains `-0.0362` behind BM25 on evidence recall@5 (`[-0.0489, -0.0237]`) and `-0.0337` behind on context sufficiency@5 (`[-0.0497, -0.0182]`). The planned expectation that dense must beat sparse therefore does not hold for either the base dense model or its reranked top 20 on this benchmark.

### Cross-benchmark sparse/dense reversal

The central Stage 1 result is not a loader caveat: the same frozen BGE-base model and harness occupy opposite extremes on two corpora. On FiQA, BGE dense beats BM25 by 0.1622 nDCG@10; on MultiHop-RAG, it trails BM25 by 0.1645 evidence recall@5. The metrics differ, so their magnitudes are not directly equated, but the winner reverses decisively.

MultiHop-RAG exposes the mechanism unusually clearly. Of 2,255 answerable questions, 2,242 name at least one gold publisher and 2,091 name every gold publisher. Those literal source names act as routing keys for BM25. This demonstrates that sparse-versus-dense performance is a property of the query/corpus relationship, not a universal model ordering. The confirmed fusion prediction is consistent with the same mechanism: combining branches strongly repairs dense retrieval, but the lexical branch remains best outright.

The reversal survives the loader checks: URL and fact mappings are complete, metadata is indexed, the [BGE query instruction](https://huggingface.co/BAAI/bge-base-en-v1.5) is present, embeddings are normalized, search is exact, and an independent implementation reproduces official-style Hits@10 for all four artifacts. BM25's advantage also appears independently in comparison, inference, and temporal questions. MultiHop-RAG's paper tested `bge-large-en-v1.5`, not the frozen `bge-base-en-v1.5`, and did not report a BM25 comparison.

On this Apple Silicon machine, measured p50 search latency was 14.6 ms/query for BM25 and 8.8 ms/query for dense. Reranking was 2,263 ms/query using cross-query batch-amortized timing, so it is not a serving-latency measurement and is not directly interchangeable with the sequential base timings. The complete rows, per-query artifacts, comparisons, and Holm-adjusted results are persisted under `results/track_b_stage1*`.

## Generation Gates

1. Oracle context: run answerable questions with annotated facts directly. If EM/F1 is poor, change the fixed generator before retrieval comparisons.
2. Determinism: repeat one frozen configuration three times. Report the spread before interpreting retriever deltas.
3. Retrieved context: vary only BM25, BGE-base dense, and BGE-base plus reranker. Use the same top-5 chunks, prompt, generator, temperature, seed, and context window.

E1 reports aggregate EM/F1 and the same paired deltas separately for comparison, inference, and temporal questions. These 24 predeclared tests form one Holm-corrected family, preventing the 29-point oracle spread across question types from being hidden by an aggregate alone.

Correctness uses frozen SQuAD-style exact match and token F1 over a concise `Answer:` field. The same frozen response contract requests a complete `Claim:` that explicitly states the answer, because an entity fragment alone is not a valid NLI hypothesis; malformed responses are preserved and flagged. Abstention always reports both null-query abstention and answerable false abstention. Faithfulness uses the [DeBERTa-v3-base NLI cross-encoder](https://huggingface.co/cross-encoder/nli-deberta-v3-base) with a threshold curve; each claim is scored against each context block and receives the maximum entailment probability, avoiding silent truncation of later chunks. A headline unsupported-claim rate requires agreement against 50 hand labels, reported with Cohen's kappa.

Retrieved prompts use a strict 1,280-word rendered-context budget, including block labels, titles, publishers, and timestamps. The body budget is balanced across the fixed top-five chunk IDs using a chunk-ID-stable allocation, so evidence-position experiments change order without also truncating the final position more aggressively. This is a whitespace-word budget, not a claim of exact Qwen-token matching.

The paper's released [`qa_evaluate.py`](https://github.com/yixuantt/MultiHop-RAG/blob/main/qa_evaluate.py) counts a prediction as correct when it shares any whitespace token with the gold answer. Its published "accuracy" values are therefore not directly comparable with this harness's stricter EM; they are contextual references only.

The final frozen prompt passed a 20-question oracle smoke test at 0.850 EM, 0.883 token F1, 1.000 response-format validity, and 0.050 false abstention. No response reached the 96-token output cap, and the longest claim contained 20 words under the 25-word contract. Because those questions were inspected while repairing the response contract, their IDs are frozen and hashed as a prompt-development slice. They are physically excluded from every full oracle and retrieved generation run, leaving 2,235 answerable evaluation questions and all 301 null questions. The smoke result selects the runnable prompt contract; it is not reported as the final oracle estimate.

The definitive held-out oracle gate reached **0.7955 EM and 0.8034 token F1** on all 2,235 evaluation questions. Performance varies materially by question type: inference is strongest at 0.9156 EM, comparison reaches 0.7976, and temporal questions are the bottleneck at 0.6252. Response-format validity is 0.9808, answerable false abstention is 0.0125, and no response reached the 96-token cap. Sequential local generation took 1.812 s/query at p50 and 2.606 s/query at p95 with short oracle evidence.

Ollama returned impossible component-duration fields on 45 of 2,235 responses even though each total request duration remained well formed. Raw values remain in the artifact; component sums exclude those rows and the summary records a 0.9799 component-ledger validity rate. Total request latency uses all rows.

The initial 20-question development probe was superseded by three repeats on a versioned SHA-256 sample of 200 held-out retrieved-context queries: 174 answerable and 26 null. Every prediction string was identical across repeats. Answerable EM was 0.6609 in every run, token F1 was 0.6648 in every run, and both observed ranges were 0.000. This is now a gate on the same BM25 long-context and answerable/null distribution used by E1, though it remains specific to the pinned model, runtime, and machine.

### Existing-artifact reanalysis and preflight controls

Response-format failure does **not** explain the oracle-to-BM25 accuracy gap. The
retrieved sample's aggregate validity of 0.885 includes null questions; among its
174 answerable questions, validity is 0.9195. Malformed responses retain a parsed
fallback answer and are not assigned zero automatically. Their EM is 0.5714,
compared with 0.6688 among well-formed responses. All 14 malformed answerable
responses merely exceeded the 25-word claim limit: eight were correct and six were
wrong. They account for only 6 of 59 answer errors (10.2%). Oracle EM conditional
on valid format is 0.7970 versus 0.7209 for its 43 malformed responses. These are
descriptive conditional rates rather than a causal estimate, but they reject the
proposed parser-loss explanation. The reproducible decomposition is
`results/track_b_format_analysis.json`.

On the same 174 answerable BM25 queries, the actual rendered context contains all
annotated evidence for 28 questions (0.1609). Of the 59 wrong answers, 50 (84.7%)
lack complete annotated evidence and 9 (15.3%) are wrong despite complete
evidence. Conversely, 96 correct answers occur without complete annotated
evidence, confirming that context sufficiency is a strict attribution diagnostic,
not an accuracy ceiling. The saved-context analysis is in
`results/track_b_failure_attribution_bm25_sample200.json`.

A no-context control on the identical 200-query ID set reaches 0.2069 EM/F1 on
the 174 answerable questions, versus BM25's 0.6609 EM and 0.6648 F1. Closed-book
inference EM is 0.000; its nonzero aggregate is concentrated in binary comparison
and temporal answers. Retrieved context therefore adds substantial answer signal,
and E1 is not rendered uninformative by model knowledge of the 2023 news corpus.
The closed-book prompt has no evidence section and explicitly permits model
knowledge, distinguishing this control from forced abstention with an empty
evidence field.

The matched 200-query E1 pilot is complete under Ollama 0.33.3:

| Retrieved context | EM | Token F1 |
| --- | ---: | ---: |
| BM25 top 5 | 0.6609 | 0.6648 |
| BGE dense top 5 | 0.6322 | 0.6442 |
| BGE dense + BGE reranker@20 top 5 | 0.6897 | 0.6951 |

None of the aggregate paired differences is resolved at this sample size. Dense
minus BM25 is -0.0287 EM with 95% CI [-0.0977, +0.0402]; reranker minus dense is
+0.0575 [-0.0115, +0.1264]; and reranker minus BM25 is +0.0287
[-0.0402, +0.0920]. No comparison among the predeclared aggregate and
question-type EM/F1 family survives Holm correction. The apparent type effects
are hypotheses, not findings: reranking raises inference EM by 0.1273 while
lowering temporal EM by 0.0816 relative to dense.

Exact-match discordance is 22.4% for BM25 versus dense, 20.7% for dense versus
reranker, and 19.0% for BM25 versus reranker. A McNemar normal approximation
using the pilot effect sizes estimates that the dense-to-reranker comparison
needs about 490 answerable queries for 80% power at two-sided alpha 0.05, or 960
at the conservative first threshold 0.05/24. Both are below the 2,235 available
answerable questions. The corresponding estimates for the smaller BM25-to-dense
effect are 2,129 and 4,168. These are exploratory planning values, not guaranteed
power. The paired results and calculation are persisted under
`results/track_b_generation_sample200_*`.

The pinned NLI scorer has been run over 2,214 non-abstaining oracle claims and 158
answerable claims from one deterministic BM25 repeat. At threshold 0.5 it labels
0.6865 of oracle claims and 0.8734 of BM25 claims unsupported. This is **not a
faithfulness result**: even the gold-evidence oracle rate fails a basic
face-validity check, and composite multi-hop claims can require several evidence
blocks while the current scorer tests each block separately. The complete
threshold curves are retained in `results/track_b_faithfulness_oracle_full.json`
and `results/track_b_faithfulness_bm25_sample200.json`; no rate is quotable until
the frozen 50-claim human calibration establishes an acceptable Cohen's kappa or
motivates a revised premise construction.

## G1 Context-Depth Sweep

### Preregistered design and prediction

Recorded before any G1 answer generation: G1 fixes the retriever at the frozen
`multihop_bm25_stem_w256_o64` run and varies retrieved depth together with the
rendered-context budget. The exact arms are k=3/768 words, k=5/1,280 words,
k=10/2,560 words, and k=20/5,120 words. The first budget is the exact value behind
the approximate 770-word design. Every arm uses balanced packing, the same
SHA-256 sample of 300 evaluation questions at seed 13, temperature zero, natural
evidence order, the fixed abstention instruction, and the same pinned generator,
prompt, parser, 8,192-token context window, and 96-token output cap.

The primary mechanism prediction is directional: k=10 and k=20 will increase
actual rendered-context sufficiency relative to k=5 because 50 of 59 errors in
the earlier BM25 sample lacked complete annotated evidence. Answer EM should
increase if missing evidence is causal, but may plateau or fall if added context
creates enough distraction; k=3 is expected to reduce sufficiency and not improve
EM. Because k and word budget move together, G1 estimates the practical effect of
"more retrieved context," not separate causal effects for depth and budget.

All results are paired and stratified over all answerable, comparison, inference,
and temporal questions. The 24 EM/F1 tests form a generation-outcome Holm family;
the 24 evidence-recall/context-sufficiency tests form a separate mechanism Holm
family. Splitting the families is preregistered because the latter measures the
retriever/context pathway rather than generated-answer quality. Raw argmax is not
treated as an unbiased test estimate.

The 300 questions are a generation-development slice. Any depth selected by G1
must be confirmed on the remaining evaluation questions using
`--exclude-evaluation-sample-size 300 --exclude-evaluation-sample-seed 13`.
This reconstructs and physically excludes the G1 IDs before a full E1 run. The
previous k=5 artifact contains only 200 questions, so it cannot serve as the paired
G1 baseline; all four 300-query arms must run.

Status at preregistration: not run. Results must be appended without editing the
design or predictions above.

**Feasibility amendment recorded before any amended run:** the original k=3,
k=5, and k=10 arms completed at the preregistered 8,192-token window. Before
starting k=20, a token-ledger probe used `mhq-0376`, the largest observed k=10
prompt. Its evaluated prompt count fell non-monotonically from 4,554 tokens at
k=10 to 4,098 at k=20, proving that Ollama had truncated the 5,120-word prompt.
The identical k=20 prompt evaluated to 8,953 tokens with a 16,384-token window.

The context window is therefore amended to 16,384 for **all four** G1 arms. The
three completed 8,192-window arms are retained for audit but excluded from G1;
rerunning every arm keeps the model configuration fixed. This amendment is based
only on prompt-token accounting, before a k=20 result or any formal G1 comparison.
The depths, word budgets, sample, predictions, and correction families above do
not change.

### Result appended after execution

The amended four-arm sweep completed on the same 300-query hash sample: 264
answerable questions and 36 null questions. Prompt-token counts are monotonic and
untruncated under the amended window, rising from a maximum of 1,657 tokens at
k=3 to 8,953 at k=20; no response reached the 96-token output cap.

| k | Rendered budget | Evidence recall | Context sufficiency | EM | Token F1 | p50 latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 768 words | 0.3494 | 0.1061 | 0.6326 | 0.6402 | 5.04 s |
| 5 | 1,280 words | 0.4495 | 0.1818 | 0.6742 | 0.6793 | 6.67 s |
| 10 | 2,560 words | 0.5893 | 0.2992 | **0.7500** | **0.7566** | 14.36 s |
| 20 | 5,120 words | **0.7263** | **0.4545** | 0.7424 | 0.7509 | 29.23 s |

The preregistered mechanism prediction is confirmed. Relative to k=5, k=10
raises evidence recall by 0.1398 with a paired 95% interval of
`[+0.1139, +0.1660]` and context sufficiency by 0.1174
`[+0.0795, +0.1591]`. At k=20, the corresponding gains are 0.2768
`[+0.2431, +0.3112]` and 0.2727 `[+0.2197, +0.3258]`. All four survive the
24-test mechanism Holm family; 23 of 24 mechanism tests survive overall.

k=10 also raises answer quality over k=5: EM increases by 0.0758
`[+0.0341, +0.1212]` and token F1 by 0.0773 `[+0.0341, +0.1212]`. These are
the only two of 24 generation-outcome tests that survive Holm correction
(`p_holm < 0.005` at 10,000-resample resolution). All three question types move
up descriptively, with the largest EM gain on temporal questions (+0.0986), but
no type-specific answer test survives the family correction.

More context stops helping answers after k=10 even though retrieval continues to
improve. The exploratory k=10-to-k=20 contrast is -0.0076 EM
`[-0.0530, +0.0379]` and -0.0057 F1 `[-0.0500, +0.0383]`, while context
sufficiency rises another 0.1553 `[+0.1136, +0.2008]`. Among 41 questions that
newly acquire complete evidence at k=20, EM moves from 0.7317 to 0.7073. This is
evidence-availability/answer-accuracy decoupling, not retrieval saturation.

Two secondary diagnostics point the same way. Overall response-format validity
falls from 0.9233 at k=10 to 0.8533 at k=20, but format failure does not cause the
answer plateau: the 27 malformed answerable k=20 responses score 0.8519 EM versus
0.7300 among well-formed responses, and the parser preserves their extracted
answers rather than assigning zero. Null-query abstention also falls from 0.8333
to 0.7500, reducing abstention separation from 0.8220 to 0.7462. These rates are
descriptive and were not part of the two confirmatory G1 families.

k=10 is the development-slice selection for confirmatory E1: it is the shallowest
arm with a family-wise significant answer gain, while k=20 offers no detectable
answer improvement and doubles p50 latency. This is not yet a held-out E1 result;
confirmation must exclude the 300 G1 IDs as specified above. The complete summary,
two preregistered families, and explicitly exploratory knee contrast are persisted
under `results/track_b_depth_*`.

## Chunking Experiment

E5 exposes three segmentation strategies behind total cache and run keys: fixed whitespace-word windows, paragraph-aware structural packing, and adjacent-sentence semantic breaks from the pinned BGE-base encoder. Semantic boundaries are persisted as a hash-verified chunk artifact and are never silently recomputed during generation.

Chunking comparisons use greedy rank-order packing under the same 1,280-word rendered-context budget, including labels and metadata. This is intentionally separate from the balanced top-five packer used by E1 and the evidence-position experiment. The planned E5 family holds BM25 and every generation control fixed and compares fixed windows without overlap, the current 64-word overlap baseline, structural chunks, and semantic chunks on the same versioned 300-query evaluation sample.
