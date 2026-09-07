from __future__ import annotations

import time
from collections.abc import Sequence
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
    runs_by_depth, latencies_by_depth = rerank_at_depths_with_latencies(
        queries=queries,
        corpus=corpus,
        base_runs=base_runs,
        model=model,
        rerank_top_k_values=[rerank_top_k],
        keep_top_k=keep_top_k,
        batch_size=batch_size,
    )
    return runs_by_depth[rerank_top_k], latencies_by_depth[rerank_top_k]


def rerank_at_depths_with_latencies(
    queries: dict[str, str],
    corpus: dict[str, dict[str, Any]],
    base_runs: dict[str, dict[str, float]],
    model: CrossEncoder,
    rerank_top_k_values: Sequence[int],
    keep_top_k: int = 1000,
    batch_size: int = 32,
) -> tuple[dict[int, dict[str, dict[str, float]]], dict[int, list[float]]]:
    depths = sorted(set(rerank_top_k_values))
    if not depths or any(depth <= 0 for depth in depths):
        raise ValueError("rerank_top_k_values must contain positive integers")

    runs_by_depth: dict[int, dict[str, dict[str, float]]] = {
        depth: {} for depth in depths
    }
    latencies_by_depth: dict[int, list[float]] = {depth: [] for depth in depths}

    for query_id, base_scores in tqdm(base_runs.items(), desc="Reranking"):
        candidates = sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))
        scored: dict[str, float] = {}
        inference_seconds = 0.0
        previous_depth = 0

        for depth in depths:
            capped_depth = min(depth, len(candidates), keep_top_k)
            new_head = candidates[previous_depth:capped_depth]
            pairs = [
                (queries[query_id], corpus_text(corpus[doc_id]))
                for doc_id, _ in new_head
            ]
            inference_started = time.perf_counter()
            if pairs:
                rerank_scores = model.predict(
                    pairs,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
            else:
                rerank_scores = []
            inference_seconds += time.perf_counter() - inference_started
            for (doc_id, _), score in zip(new_head, rerank_scores, strict=True):
                scored[doc_id] = float(score)

            build_started = time.perf_counter()
            runs_by_depth[depth][query_id] = _build_reranked_run(
                candidates=candidates,
                rerank_scores=scored,
                rerank_top_k=capped_depth,
                keep_top_k=keep_top_k,
            )
            latencies_by_depth[depth].append(
                inference_seconds + (time.perf_counter() - build_started)
            )
            previous_depth = capped_depth

    return runs_by_depth, latencies_by_depth


def _build_reranked_run(
    candidates: list[tuple[str, float]],
    rerank_scores: dict[str, float],
    rerank_top_k: int,
    keep_top_k: int,
) -> dict[str, float]:
    head = candidates[:rerank_top_k]
    tail = candidates[rerank_top_k:keep_top_k]
    scores = {doc_id: rerank_scores[doc_id] for doc_id, _ in head}
    if tail:
        min_rerank_score = min(scores.values()) if scores else 0.0
        for rank, (doc_id, _) in enumerate(tail, start=1):
            scores[doc_id] = min_rerank_score - rank
    return dict(
        sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:keep_top_k]
    )
