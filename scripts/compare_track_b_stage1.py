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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Track B Stage 1 evidence runs.")
    parser.add_argument("--runs", default="results/track_b_stage1.csv")
    parser.add_argument("--output", default="results/track_b_stage1_comparisons.csv")
    parser.add_argument(
        "--holm-output", default="results/track_b_stage1_comparisons_holm.csv"
    )
    parser.add_argument(
        "--metrics", default="evidence_recall@5,context_sufficiency@5"
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = read_rows(Path(args.runs))
    by_type = {row["retriever_type"]: row for row in rows}
    pairs = [
        ("bm25_to_bge", by_type["bm25"], by_type["dense"]),
        ("bge_to_bge_rerank20", by_type["dense"], by_type["rerank"]),
        ("bm25_to_bge_rerank20", by_type["bm25"], by_type["rerank"]),
    ]
    result_rows: list[dict[str, Any]] = []
    for metric in [item.strip() for item in args.metrics.split(",") if item.strip()]:
        for name, baseline, candidate in pairs:
            result = compare(
                baseline_path=Path(baseline["per_query_metrics_path"]),
                candidate_path=Path(candidate["per_query_metrics_path"]),
                baseline_run_id=baseline["run_id"],
                candidate_run_id=candidate["run_id"],
                metric=metric,
                samples=args.samples,
                seed=args.seed,
            )
            result_rows.append(
                {
                    "comparison": name,
                    "baseline_run_name": baseline["run_name"],
                    "candidate_run_name": candidate["run_name"],
                    **result,
                }
            )
    write_rows(Path(args.output), result_rows)
    adjust_comparisons(Path(args.output), Path(args.holm_output))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


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
