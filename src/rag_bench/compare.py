from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rag_bench.stats import metric_delta, paired_bootstrap_ci, paired_bootstrap_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two per-query metric artifacts.")
    parser.add_argument("--baseline", required=True, help="Baseline per-query CSV artifact.")
    parser.add_argument("--candidate", required=True, help="Candidate per-query CSV artifact.")
    parser.add_argument("--baseline-run-id", help="Baseline run ID. Defaults to deriving it from the artifact filename.")
    parser.add_argument("--candidate-run-id", help="Candidate run ID. Defaults to deriving it from the artifact filename.")
    parser.add_argument("--metric", default="ndcg@10", help="Metric column to compare.")
    parser.add_argument("--samples", type=int, default=1000, help="Bootstrap sample count.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level.")
    parser.add_argument("--seed", type=int, default=13, help="Bootstrap RNG seed.")
    parser.add_argument("--output", help="Optional CSV file to append the comparison row.")
    args = parser.parse_args()

    row = compare(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        baseline_run_id=args.baseline_run_id,
        candidate_run_id=args.candidate_run_id,
        metric=args.metric,
        samples=args.samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    if args.output:
        append_comparison(Path(args.output), row)
    print(json.dumps(row, indent=2, sort_keys=True))


def compare(
    baseline_path: Path,
    candidate_path: Path,
    metric: str,
    baseline_run_id: str | None = None,
    candidate_run_id: str | None = None,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 13,
) -> dict[str, Any]:
    baseline = read_per_query_metric(baseline_path, metric)
    candidate = read_per_query_metric(candidate_path, metric)
    if set(baseline) != set(candidate):
        missing_from_baseline = sorted(set(candidate) - set(baseline))
        missing_from_candidate = sorted(set(baseline) - set(candidate))
        raise ValueError(
            "Per-query files must contain identical query IDs for paired comparison. "
            f"missing_from_baseline={missing_from_baseline[:5]}, "
            f"missing_from_candidate={missing_from_candidate[:5]}"
        )
    query_ids = sorted(baseline)

    def scorer(sampled_query_ids: list[str]) -> float:
        return metric_delta(baseline, candidate, sampled_query_ids)

    ci_low, ci_high = paired_bootstrap_ci(
        query_ids=query_ids,
        scorer=scorer,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    bootstrap_scores = paired_bootstrap_scores(
        query_ids=query_ids,
        scorer=scorer,
        samples=samples,
        seed=seed,
    )
    baseline_mean = sum(baseline[query_id] for query_id in query_ids) / len(query_ids)
    candidate_mean = sum(candidate[query_id] for query_id in query_ids) / len(query_ids)
    delta = candidate_mean - baseline_mean
    p_two_sided = bootstrap_two_sided_p_value(bootstrap_scores, delta)

    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline_run_id": baseline_run_id or run_id_from_artifact_path(baseline_path),
        "candidate_run_id": candidate_run_id or run_id_from_artifact_path(candidate_path),
        "metric": metric,
        "num_queries": len(query_ids),
        "baseline_mean": round(baseline_mean, 6),
        "candidate_mean": round(candidate_mean, 6),
        "delta": round(delta, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "p_two_sided": round(p_two_sided, 6),
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
    }


def read_per_query_metric(path: Path, metric: str) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(f"{path} does not contain metric column {metric!r}.")
        return {row["query_id"]: float(row[metric]) for row in reader}


def append_comparison(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    existing_rows: list[dict[str, str]] = []
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            old_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        fieldnames = old_fieldnames + [field for field in fieldnames if field not in old_fieldnames]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)


def run_id_from_artifact_path(path: Path) -> str:
    name = path.name
    for suffix in (".per_query.csv", ".run.jsonl", ".metadata.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def bootstrap_two_sided_p_value(scores, observed_delta: float) -> float:
    if len(scores) == 0:
        return 1.0
    if observed_delta >= 0:
        one_sided = sum(score <= 0 for score in scores) / len(scores)
    else:
        one_sided = sum(score >= 0 for score in scores) / len(scores)
    return min(1.0, 2.0 * one_sided)


if __name__ == "__main__":
    main()
