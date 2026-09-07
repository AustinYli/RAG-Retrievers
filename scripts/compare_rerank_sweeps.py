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
        description="Compare persisted reranker sweeps against their base retrieval runs."
    )
    parser.add_argument("--runs", default="results/track_a_runs.csv")
    parser.add_argument("--rerank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holm-output", required=True)
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--include-adjacent-depths",
        action="store_true",
        help="Also compare each depth with the next larger depth for the same base/model.",
    )
    args = parser.parse_args()

    base_rows = rows_by_id(Path(args.runs))
    rerank_rows = latest_rows_by_run_name(Path(args.rerank))
    comparisons: list[dict[str, Any]] = []
    sorted_rerank_rows = sorted(
        rerank_rows.values(),
        key=lambda row: (
            row["dataset"],
            row["base_run_name"],
            int(row["rerank_top_k"]),
        ),
    )
    for rerank_row in sorted_rerank_rows:
        base_row = base_rows[rerank_row["base_run_id"]]
        candidate_path = Path(rerank_row.get("per_query_metrics_path", ""))
        if not candidate_path.is_file():
            continue
        comparison = compare(
            baseline_path=Path(base_row["per_query_metrics_path"]),
            candidate_path=candidate_path,
            baseline_run_id=base_row["run_id"],
            candidate_run_id=rerank_row["run_name"],
            metric=args.metric,
            samples=args.samples,
            seed=args.seed,
        )
        comparisons.append(
            {
                "comparison": "base_to_reranker",
                "dataset": rerank_row["dataset"],
                "base_run_name": rerank_row["base_run_name"],
                "candidate_run_name": rerank_row["run_name"],
                "reranker_model": rerank_row["reranker_model"],
                "rerank_top_k": rerank_row["rerank_top_k"],
                "device": rerank_row.get("device", ""),
                "rerank_latency_p50_ms": rerank_row.get("rerank_latency_p50_ms", ""),
                "rerank_latency_p95_ms": rerank_row.get("rerank_latency_p95_ms", ""),
                **comparison,
            }
        )

    if args.include_adjacent_depths:
        grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in sorted_rerank_rows:
            if Path(row.get("per_query_metrics_path", "")).is_file():
                key = (row["dataset"], row["base_run_id"], row["reranker_model"])
                grouped.setdefault(key, []).append(row)
        for rows in grouped.values():
            ordered = sorted(rows, key=lambda row: int(row["rerank_top_k"]))
            for baseline_row, candidate_row in zip(ordered, ordered[1:]):
                comparison = compare(
                    baseline_path=Path(baseline_row["per_query_metrics_path"]),
                    candidate_path=Path(candidate_row["per_query_metrics_path"]),
                    baseline_run_id=baseline_row["run_name"],
                    candidate_run_id=candidate_row["run_name"],
                    metric=args.metric,
                    samples=args.samples,
                    seed=args.seed,
                )
                comparisons.append(
                    {
                        "comparison": "rerank_depth_to_next_depth",
                        "dataset": candidate_row["dataset"],
                        "base_run_name": candidate_row["base_run_name"],
                        "candidate_run_name": candidate_row["run_name"],
                        "reranker_model": candidate_row["reranker_model"],
                        "baseline_rerank_top_k": baseline_row["rerank_top_k"],
                        "rerank_top_k": candidate_row["rerank_top_k"],
                        "device": candidate_row.get("device", ""),
                        "rerank_latency_p50_ms": candidate_row.get(
                            "rerank_latency_p50_ms", ""
                        ),
                        "rerank_latency_p95_ms": candidate_row.get(
                            "rerank_latency_p95_ms", ""
                        ),
                        **comparison,
                    }
                )

    write_rows(Path(args.output), comparisons)
    adjust_comparisons(Path(args.output), Path(args.holm_output))


def rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["run_id"]: row for row in csv.DictReader(file)}


def latest_rows_by_run_name(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["run_name"]: row for row in csv.DictReader(file)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(field for field in row if field not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
