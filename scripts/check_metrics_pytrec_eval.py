from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytrec_eval

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_bench.load import load_beir_dataset
from rag_bench.metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check persisted nDCG, MAP, and recall against pytrec_eval."
    )
    parser.add_argument("--runs", default="results/track_a_runs.csv")
    parser.add_argument("--output", default="results/pytrec_eval_crosscheck.csv")
    parser.add_argument("--cutoffs", default="1,5,10,100")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    cutoffs = [int(value) for value in args.cutoffs.split(",")]
    rows = read_rows(Path(args.runs))
    qrels_by_dataset: dict[str, dict[str, dict[str, int]]] = {}
    checks: list[dict[str, Any]] = []
    for row in rows:
        dataset = row["dataset"]
        if dataset not in qrels_by_dataset:
            _, _, qrels_by_dataset[dataset] = load_beir_dataset(
                dataset=dataset,
                split=row.get("split") or "test",
                data_dir="data/beir",
            )
        qrels = qrels_by_dataset[dataset]
        run = read_run(Path(row["run_artifact_path"]))
        ours = evaluate(qrels, run, cutoffs)
        rank_normalized_run = rank_normalize(run)
        measures = {
            *(f"ndcg_cut.{cutoff}" for cutoff in cutoffs),
            *(f"map_cut.{cutoff}" for cutoff in cutoffs),
            *(f"recall.{cutoff}" for cutoff in cutoffs),
        }
        theirs_per_query = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(
            rank_normalized_run
        )
        for family, pytrec_name in (
            ("ndcg", "ndcg_cut"),
            ("map", "map_cut"),
            ("recall", "recall"),
        ):
            for cutoff in cutoffs:
                metric = f"{family}@{cutoff}"
                pytrec_key = f"{pytrec_name}_{cutoff}"
                theirs = sum(
                    theirs_per_query.get(query_id, {}).get(pytrec_key, 0.0)
                    for query_id in qrels
                ) / len(qrels)
                difference = ours[metric] - theirs
                checks.append(
                    {
                        "run_id": row["run_id"],
                        "run_name": row["run_name"],
                        "dataset": dataset,
                        "metric": metric,
                        "comparison_basis": "harness_rank_with_strict_surrogate_scores",
                        "harness_value": ours[metric],
                        "pytrec_eval_value": theirs,
                        "absolute_difference": abs(difference),
                        "within_tolerance": abs(difference) <= args.tolerance,
                    }
                )

    write_rows(Path(args.output), checks)
    failures = [row for row in checks if not row["within_tolerance"]]
    print(
        json.dumps(
            {
                "checks": len(checks),
                "failures": len(failures),
                "max_absolute_difference": max(
                    (float(row["absolute_difference"]) for row in checks),
                    default=0.0,
                ),
                "tolerance": args.tolerance,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            run.setdefault(item["query_id"], {})[item["doc_id"]] = float(item["score"])
    return run


def rank_normalize(
    run: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for query_id, scores in run.items():
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        normalized[query_id] = {
            doc_id: float(len(ranked) - rank) for rank, (doc_id, _) in enumerate(ranked)
        }
    return normalized


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
