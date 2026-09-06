from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from rag_bench.compare import compare

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adjust_comparisons import adjust_comparisons


MODEL_LABELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "minilm",
    "BAAI/bge-base-en-v1.5": "bge",
    "intfloat/e5-base-v2": "e5",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired bootstrap comparisons for Track A.")
    parser.add_argument("--runs", default="results/track_a_runs.csv")
    parser.add_argument("--output", default="results/track_a_comparisons.csv")
    parser.add_argument("--holm-output", default="results/track_a_comparisons_holm.csv")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = latest_rows_by_run_name(Path(args.runs))
    comparisons = build_comparisons(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    for comparison_name, baseline, candidate in comparisons:
        row = compare(
            baseline_path=Path(baseline["per_query_metrics_path"]),
            candidate_path=Path(candidate["per_query_metrics_path"]),
            baseline_run_id=baseline["run_id"],
            candidate_run_id=candidate["run_id"],
            metric=args.metric,
            samples=args.samples,
            seed=args.seed,
        )
        row = {
            "comparison": comparison_name,
            "dataset": baseline["dataset"],
            "baseline_run_name": baseline["run_name"],
            "candidate_run_name": candidate["run_name"],
            **row,
        }
        result_rows.append(row)

    write_rows(output_path, result_rows)
    adjust_comparisons(output_path, Path(args.holm_output))


def latest_rows_by_run_name(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows[row["run_name"]] = row
    return rows


def build_comparisons(rows: dict[str, dict[str, str]]) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    comparisons: list[tuple[str, dict[str, str], dict[str, str]]] = []
    datasets = sorted({row["dataset"] for row in rows.values()})
    for dataset in datasets:
        bm25 = rows[f"{dataset}_bm25_stem_512"]
        dense_rows = typed_rows(rows, dataset, "dense")
        hybrid_rows = typed_rows(rows, dataset, "hybrid")
        best_dense = max(dense_rows, key=lambda row: float(row["ndcg@10"]))
        best_hybrid = max(hybrid_rows, key=lambda row: float(row["ndcg@10"]))

        comparisons.append(("bm25_to_best_dense", bm25, best_dense))
        comparisons.append(("bm25_to_best_hybrid", bm25, best_hybrid))
        comparisons.append(("best_dense_to_best_hybrid", best_dense, best_hybrid))

        hybrids_by_model = {row["dense_model"]: row for row in hybrid_rows}
        for dense in sorted(dense_rows, key=lambda row: model_label(row)):
            label = model_label(dense)
            hybrid = hybrids_by_model[dense["dense_model"]]
            comparisons.append((f"bm25_to_dense_{label}", bm25, dense))
            comparisons.append((f"bm25_to_hybrid_{label}", bm25, hybrid))
            comparisons.append((f"dense_to_hybrid_{label}", dense, hybrid))
    return dedupe_comparisons(comparisons)


def typed_rows(
    rows: dict[str, dict[str, str]], dataset: str, retriever_type: str
) -> list[dict[str, str]]:
    return [
        row
        for row in rows.values()
        if row.get("dataset") == dataset and row.get("retriever_type") == retriever_type
    ]


def model_label(row: dict[str, str]) -> str:
    return MODEL_LABELS.get(row["dense_model"], row["dense_model"].replace("/", "_").lower())


def dedupe_comparisons(
    comparisons: list[tuple[str, dict[str, str], dict[str, str]]],
) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for name, baseline, candidate in comparisons:
        key = (baseline["dataset"], baseline["run_id"], candidate["run_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, baseline, candidate))
    return unique


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(field for field in row if field not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
