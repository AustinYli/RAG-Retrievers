from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from rag_bench.multihop import load_multihop_dataset


METRICS = ("evidence_recall@5", "context_sufficiency@5", "evidence_hit@10")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratify Track B retrieval metrics by question type."
    )
    parser.add_argument("--runs", default="results/track_b_stage1.csv")
    parser.add_argument("--data-dir", default="data/multihop_rag")
    parser.add_argument("--output", default="results/track_b_stage1_by_type.csv")
    args = parser.parse_args()

    dataset = load_multihop_dataset(args.data_dir)
    with Path(args.runs).open("r", newline="", encoding="utf-8") as file:
        run_rows = list(csv.DictReader(file))

    output_rows: list[dict[str, Any]] = []
    for run_row in run_rows:
        per_query = read_rows(Path(run_row["per_query_metrics_path"]))
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        grouped["all_answerable"] = list(per_query.values())
        for query_id, row in per_query.items():
            grouped[dataset.question_types[query_id]].append(row)
        for question_type, rows in sorted(grouped.items()):
            output_rows.append(
                {
                    "run_id": run_row["run_id"],
                    "run_name": run_row["run_name"],
                    "retriever_type": run_row["retriever_type"],
                    "question_type": question_type,
                    "num_queries": len(rows),
                    **{
                        metric: round(mean(float(row[metric]) for row in rows), 6)
                        for metric in METRICS
                    },
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["query_id"]: row for row in csv.DictReader(file)}


if __name__ == "__main__":
    main()
