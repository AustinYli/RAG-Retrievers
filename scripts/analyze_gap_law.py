from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the association between branch quality gap and equal-weight RRF benefit."
    )
    parser.add_argument("--runs", default="results/track_a_runs.csv")
    parser.add_argument("--output", default="results/gap_law.csv")
    parser.add_argument("--summary", default="results/gap_law_summary.json")
    args = parser.parse_args()

    rows = latest_rows_by_run_name(Path(args.runs))
    observations = build_observations(rows)
    write_rows(Path(args.output), observations)

    gaps = [float(row["branch_gap_dense_minus_bm25"]) for row in observations]
    benefits = [float(row["fusion_benefit_hybrid_minus_dense"]) for row in observations]
    summary = {
        "num_observations": len(observations),
        "pearson_r": round(pearson(gaps, benefits), 6),
        "spearman_rho": round(pearson(ranks(gaps), ranks(benefits)), 6),
        "ols_slope": round(ols_slope(gaps, benefits), 6),
        "scope": "Post-hoc association across 3 BEIR datasets x 3 dense models; not a validated threshold.",
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_observations(rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    bm25_by_dataset = {
        row["dataset"]: row
        for row in rows.values()
        if row.get("retriever_type") == "bm25"
    }
    hybrid_by_key = {
        (row["dataset"], row["dense_model"]): row
        for row in rows.values()
        if row.get("retriever_type") == "hybrid"
    }
    observations: list[dict[str, Any]] = []
    for dense in rows.values():
        if dense.get("retriever_type") != "dense":
            continue
        bm25 = bm25_by_dataset[dense["dataset"]]
        hybrid = hybrid_by_key[(dense["dataset"], dense["dense_model"])]
        bm25_score = float(bm25["ndcg@10"])
        dense_score = float(dense["ndcg@10"])
        hybrid_score = float(hybrid["ndcg@10"])
        observations.append(
            {
                "dataset": dense["dataset"],
                "dense_model": dense["dense_model"],
                "bm25_ndcg@10": bm25_score,
                "dense_ndcg@10": dense_score,
                "hybrid_ndcg@10": hybrid_score,
                "branch_gap_dense_minus_bm25": round(dense_score - bm25_score, 6),
                "fusion_benefit_hybrid_minus_dense": round(
                    hybrid_score - dense_score, 6
                ),
            }
        )
    return sorted(
        observations, key=lambda row: float(row["branch_gap_dense_minus_bm25"])
    )


def latest_rows_by_run_name(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["run_name"]: row for row in csv.DictReader(file)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Inputs must have the same nonzero length")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for original_index, _ in ordered[index:end]:
            result[original_index] = average_rank
        index = end
    return result


def ols_slope(x_values: list[float], y_values: list[float]) -> float:
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
