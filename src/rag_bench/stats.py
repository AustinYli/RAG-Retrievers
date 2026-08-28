from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


def paired_bootstrap_ci(
    query_ids: list[str],
    scorer: Callable[[list[str]], float],
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 13,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval over query-level paired samples."""
    scores = paired_bootstrap_scores(query_ids, scorer, samples=samples, seed=seed)
    if scores.size == 0:
        return 0.0, 0.0

    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1.0 - alpha))


def paired_bootstrap_scores(
    query_ids: list[str],
    scorer: Callable[[list[str]], float],
    samples: int = 1000,
    seed: int = 13,
) -> np.ndarray:
    """Return bootstrap replicate scores over query-level paired samples."""
    if not query_ids:
        return np.array([], dtype=float)

    rng = np.random.default_rng(seed)
    scores = []
    query_ids_array = np.array(query_ids)
    for _ in range(samples):
        sample = rng.choice(query_ids_array, size=len(query_ids_array), replace=True).tolist()
        scores.append(scorer(sample))
    return np.array(scores, dtype=float)


def metric_delta(
    metric_by_query_a: Mapping[str, float],
    metric_by_query_b: Mapping[str, float],
    query_ids: list[str],
) -> float:
    if not query_ids:
        return 0.0
    missing_a = [query_id for query_id in query_ids if query_id not in metric_by_query_a]
    missing_b = [query_id for query_id in query_ids if query_id not in metric_by_query_b]
    if missing_a or missing_b:
        raise KeyError(
            f"Missing query IDs for paired delta: baseline={missing_a[:5]}, candidate={missing_b[:5]}"
        )
    return float(
        np.mean([metric_by_query_b[query_id] - metric_by_query_a[query_id] for query_id in query_ids])
    )
