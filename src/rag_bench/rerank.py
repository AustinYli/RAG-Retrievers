from __future__ import annotations

import time
from typing import Any

from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

from rag_bench.load import corpus_text


def rerank(
    queries: dict[str, str],
    corpus: dict[str, dict[str, Any]],
    base_runs: dict[str, dict[str, float]],
    model_name: str,
    rerank_top_k: int = 100,
    keep_top_k: int = 1000,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, dict[str, float]]:
    model = CrossEncoder(model_name, device=device)
    runs, _ = rerank_with_latencies(
        queries=queries,
        corpus=corpus,
        base_runs=base_runs,
        model=model,
        rerank_top_k=rerank_top_k,
        keep_top_k=keep_top_k,
        batch_size=batch_size,
    )
    return runs


def rerank_with_latencies(
    queries: dict[str, str],
    corpus: dict[str, dict[str, Any]],
    base_runs: dict[str, dict[str, float]],
    model: CrossEncoder,
    rerank_top_k: int = 100,
    keep_top_k: int = 1000,
    batch_size: int = 32,
) -> tuple[dict[str, dict[str, float]], list[float]]:
    reranked_runs: dict[str, dict[str, float]] = {}
    latencies: list[float] = []

    for query_id, base_scores in tqdm(base_runs.items(), desc="Reranking"):
        started = time.perf_counter()
        candidates = sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))
        head = candidates[:rerank_top_k]
        tail = candidates[rerank_top_k:keep_top_k]

        pairs = [(queries[query_id], corpus_text(corpus[doc_id])) for doc_id, _ in head]
        if pairs:
            rerank_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        else:
            rerank_scores = []

        scores: dict[str, float] = {}
        for (doc_id, _), score in zip(head, rerank_scores, strict=False):
            scores[doc_id] = float(score)

        if tail:
            min_rerank_score = min(scores.values()) if scores else 0.0
            for rank, (doc_id, _) in enumerate(tail, start=1):
                scores[doc_id] = min_rerank_score - rank

        reranked_runs[query_id] = dict(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:keep_top_k]
        )
        latencies.append(time.perf_counter() - started)

    return reranked_runs, latencies
