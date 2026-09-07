from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


Run = dict[str, dict[str, float]]
Qrels = Mapping[str, Mapping[str, int]]


def evaluate(qrels: Qrels, runs: Run, cutoffs: Sequence[int]) -> dict[str, float]:
    per_query = evaluate_per_query(qrels, runs, cutoffs)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        for name in ("ndcg", "recall", "mrr", "map"):
            metric_name = f"{name}@{cutoff}"
            metrics[metric_name] = _mean(
                [query_metrics[metric_name] for query_metrics in per_query.values()]
            )
    return metrics


def evaluate_per_query(
    qrels: Qrels,
    runs: Run,
    cutoffs: Sequence[int],
) -> dict[str, dict[str, float]]:
    per_query: dict[str, dict[str, float]] = {}
    for query_id, rels in qrels.items():
        per_query[query_id] = {}
        for cutoff in cutoffs:
            ranked = _ranked_doc_ids(runs.get(query_id, {}), cutoff)
            per_query[query_id][f"ndcg@{cutoff}"] = _ndcg(rels, ranked, cutoff)
            per_query[query_id][f"recall@{cutoff}"] = _recall(rels, ranked)
            per_query[query_id][f"mrr@{cutoff}"] = _mrr(rels, ranked)
            per_query[query_id][f"map@{cutoff}"] = _ap(rels, ranked)
    return per_query


def _ranked_doc_ids(scores: Mapping[str, float], cutoff: int) -> list[str]:
    return [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
            :cutoff
        ]
    ]


def _ndcg(rels: Mapping[str, int], ranked: Sequence[str], cutoff: int) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(ranked, start=1):
        gain = rels.get(doc_id, 0)
        if gain > 0:
            dcg += gain / math.log2(index + 1)

    ideal_gains = sorted([gain for gain in rels.values() if gain > 0], reverse=True)[
        :cutoff
    ]
    idcg = sum(
        gain / math.log2(index + 1) for index, gain in enumerate(ideal_gains, start=1)
    )
    return dcg / idcg if idcg else 0.0


def _recall(rels: Mapping[str, int], ranked: Sequence[str]) -> float:
    relevant = {doc_id for doc_id, gain in rels.items() if gain > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked)
    return len(relevant & retrieved) / len(relevant)


def _mrr(rels: Mapping[str, int], ranked: Sequence[str]) -> float:
    for index, doc_id in enumerate(ranked, start=1):
        if rels.get(doc_id, 0) > 0:
            return 1.0 / index
    return 0.0


def _ap(rels: Mapping[str, int], ranked: Sequence[str]) -> float:
    relevant = {doc_id for doc_id, gain in rels.items() if gain > 0}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / index
    return precision_sum / len(relevant)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
