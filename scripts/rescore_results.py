from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_bench.load import load_beir_dataset
from rag_bench.metrics import evaluate, evaluate_per_query


METRIC_PATTERN = re.compile(r"^(?:ndcg|recall|mrr|map)@(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute aggregate and per-query metrics from persisted run artifacts."
    )
    parser.add_argument(
        "results", nargs="+", help="Result CSV files to update in place."
    )
    parser.add_argument("--data-dir", default="data/beir")
    args = parser.parse_args()

    for result_name in args.results:
        updated, skipped = rescore_file(Path(result_name), Path(args.data_dir))
        print(
            f"{result_name}: rescored {updated} row(s), skipped {skipped} without artifacts"
        )


def rescore_file(path: Path, data_dir: Path) -> tuple[int, int]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    cutoffs = sorted(
        {
            int(match.group(1))
            for field in fieldnames
            if (match := METRIC_PATTERN.match(field))
        }
    )
    if not cutoffs:
        raise ValueError(f"No supported metric columns found in {path}")

    qrels_by_dataset: dict[str, dict[str, dict[str, int]]] = {}
    updated = 0
    skipped = 0
    for row in rows:
        run_artifact = row.get("run_artifact_path", "")
        per_query_artifact = row.get("per_query_metrics_path", "")
        if not run_artifact or not per_query_artifact:
            skipped += 1
            continue
        run_path = Path(run_artifact)
        per_query_path = Path(per_query_artifact)
        if not run_path.is_file():
            skipped += 1
            continue

        dataset = row["dataset"]
        if dataset not in qrels_by_dataset:
            _, _, qrels_by_dataset[dataset] = load_beir_dataset(
                dataset=dataset,
                split=row.get("split") or "test",
                data_dir=str(data_dir),
            )
        run = read_run(run_path)
        metrics = evaluate(qrels_by_dataset[dataset], run, cutoffs)
        per_query = evaluate_per_query(qrels_by_dataset[dataset], run, cutoffs)
        for metric, value in metrics.items():
            if metric in fieldnames:
                row[metric] = str(round(value, 6))
        write_per_query(per_query_path, per_query)
        updated += 1

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return updated, skipped


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            run.setdefault(item["query_id"], {})[item["doc_id"]] = float(item["score"])
    return run


def write_per_query(path: Path, per_query: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = sorted(
        {metric for metrics in per_query.values() for metric in metrics}
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["query_id", *metric_fields], lineterminator="\n"
        )
        writer.writeheader()
        for query_id in sorted(per_query):
            writer.writerow({"query_id": query_id, **per_query[query_id]})


if __name__ == "__main__":
    main()
