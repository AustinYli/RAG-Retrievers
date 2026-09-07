from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_bench.compare import compare
from scripts.adjust_comparisons import adjust_comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare each dataset's best weighted-fusion row against its best dense baseline."
    )
    parser.add_argument("--runs", default="results/track_a_runs.csv")
    parser.add_argument("--fusion", default="results/fusion_sweeps.csv")
    parser.add_argument("--output", default="results/fusion_winner_comparisons.csv")
    parser.add_argument(
        "--holm-output", default="results/fusion_winner_comparisons_holm.csv"
    )
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    runs = latest_rows_by_run_name(Path(args.runs))
    fusion_rows = latest_rows_by_run_name(Path(args.fusion))
    result_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in fusion_rows.values()}):
        dense = best_dense_row(runs, dataset)
        weighted = best_weighted_fusion_row(fusion_rows, dataset)
        row = compare(
            baseline_path=Path(dense["per_query_metrics_path"]),
            candidate_path=Path(weighted["per_query_metrics_path"]),
            baseline_run_id=dense["run_id"],
            candidate_run_id=weighted["run_name"],
            metric=args.metric,
            samples=args.samples,
            seed=args.seed,
        )
        row = {
            "comparison": "best_dense_to_posthoc_best_weighted_fusion",
            "dataset": dataset,
            "selection_scope": "argmax_over_dense_weight_sweep_on_same_test_queries",
            "baseline_run_name": dense["run_name"],
            "candidate_run_name": weighted["run_name"],
            "candidate_rrf_k": weighted["rrf_k"],
            "candidate_bm25_weight": weighted["bm25_weight"],
            "candidate_dense_weight": weighted["dense_weight"],
            **row,
        }
        result_rows.append(row)

    write_rows(Path(args.output), result_rows)
    adjust_comparisons(Path(args.output), Path(args.holm_output))


def latest_rows_by_run_name(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows[row["run_name"]] = row
    return rows


def best_dense_row(rows: dict[str, dict[str, str]], dataset: str) -> dict[str, str]:
    candidates = [
        row
        for row in rows.values()
        if row.get("dataset") == dataset and row.get("retriever_type") == "dense"
    ]
    return max(candidates, key=lambda row: float(row["ndcg@10"]))


def best_weighted_fusion_row(
    rows: dict[str, dict[str, str]], dataset: str
) -> dict[str, str]:
    candidates = [
        row
        for row in rows.values()
        if row.get("dataset") == dataset and row.get("sweep_name") == "dense_weight"
    ]
    return max(candidates, key=lambda row: float(row["ndcg@10"]))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(field for field in row if field not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
