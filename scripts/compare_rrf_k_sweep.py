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
        description="Compare each best post-hoc equal-weight RRF setting with fixed k=60."
    )
    parser.add_argument("--fusion", default="results/fusion_sweeps.csv")
    parser.add_argument("--output", default="results/rrf_k_comparisons.csv")
    parser.add_argument("--holm-output", default="results/rrf_k_comparisons_holm.csv")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = read_rows(Path(args.fusion))
    comparisons: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        candidates = [
            row
            for row in rows
            if row["dataset"] == dataset and row["sweep_name"] == "rrf_k"
        ]
        baseline = next(row for row in candidates if float(row["rrf_k"]) == 60)
        candidate = max(candidates, key=lambda row: float(row[args.metric]))
        comparison = compare(
            baseline_path=Path(baseline["per_query_metrics_path"]),
            candidate_path=Path(candidate["per_query_metrics_path"]),
            baseline_run_id=baseline["run_name"],
            candidate_run_id=candidate["run_name"],
            metric=args.metric,
            samples=args.samples,
            seed=args.seed,
        )
        comparisons.append(
            {
                "comparison": "fixed_rrf60_to_posthoc_best_rrf_k",
                "dataset": dataset,
                "selection_scope": "argmax_over_rrf_k_sweep_on_same_test_queries",
                "baseline_rrf_k": baseline["rrf_k"],
                "candidate_rrf_k": candidate["rrf_k"],
                **comparison,
            }
        )

    write_rows(Path(args.output), comparisons)
    adjust_comparisons(Path(args.output), Path(args.holm_output))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
