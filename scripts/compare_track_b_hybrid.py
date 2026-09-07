from __future__ import annotations

import argparse
import csv
import json
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
        description="Evaluate the preregistered Track B hybrid prediction."
    )
    parser.add_argument("--runs", default="results/track_b_stage1.csv")
    parser.add_argument("--gap-law", default="results/gap_law.csv")
    parser.add_argument(
        "--output", default="results/track_b_hybrid_comparisons.csv"
    )
    parser.add_argument(
        "--holm-output", default="results/track_b_hybrid_comparisons_holm.csv"
    )
    parser.add_argument(
        "--prediction-output", default="results/track_b_gap_prediction.json"
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    by_type = read_latest_by_type(Path(args.runs))
    required = {"bm25", "dense", "hybrid"}
    missing = sorted(required - by_type.keys())
    if missing:
        raise ValueError(f"Missing required Track B retrieval rows: {missing}")
    validate_hybrid_config(by_type["hybrid"])

    pairs = (
        ("bge_to_frozen_hybrid", by_type["dense"], by_type["hybrid"]),
        ("bm25_to_frozen_hybrid", by_type["bm25"], by_type["hybrid"]),
    )
    comparisons: list[dict[str, Any]] = []
    for metric in ("evidence_recall@5", "context_sufficiency@5"):
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
            comparisons.append(
                {
                    "comparison": name,
                    "baseline_run_name": baseline["run_name"],
                    "candidate_run_name": candidate["run_name"],
                    **result,
                }
            )

    write_rows(Path(args.output), comparisons)
    adjust_comparisons(Path(args.output), Path(args.holm_output))
    primary = next(
        row
        for row in comparisons
        if row["comparison"] == "bge_to_frozen_hybrid"
        and row["metric"] == "evidence_recall@5"
    )
    prior_benefits = read_prior_fusion_benefits(Path(args.gap_law))
    prediction = evaluate_prediction(float(primary["delta"]), prior_benefits)
    prediction.update(
        {
            "preregistered_claim": (
                "MultiHop-RAG hybrid-minus-dense evidence recall@5 ranks above "
                "all nine Track A hybrid-minus-dense nDCG@10 gains."
            ),
            "track_b_metric": "evidence_recall@5",
            "track_a_metric": "nDCG@10",
            "track_b_hybrid_minus_dense": primary["delta"],
            "track_b_ci_low": primary["ci_low"],
            "track_b_ci_high": primary["ci_high"],
            "track_b_dense_minus_bm25": round(
                float(by_type["dense"]["evidence_recall@5"])
                - float(by_type["bm25"]["evidence_recall@5"]),
                6,
            ),
            "frozen_rrf_k": int(float(by_type["hybrid"]["rrf_k"])),
            "frozen_bm25_weight": float(by_type["hybrid"]["bm25_weight"]),
            "frozen_dense_weight": float(by_type["hybrid"]["dense_weight"]),
            "comparability_caveat": (
                "This is a directional rank-order test across different metrics; "
                "it is not a transferred point estimate."
            ),
        }
    )
    prediction_path = Path(args.prediction_output)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(
        json.dumps(prediction, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(prediction, indent=2, sort_keys=True))


def validate_hybrid_config(row: dict[str, str]) -> None:
    expected = {"rrf_k": 60.0, "bm25_weight": 1.0, "dense_weight": 1.0}
    mismatched = {
        field: row.get(field)
        for field, value in expected.items()
        if float(row.get(field) or "nan") != value
    }
    if mismatched:
        raise ValueError(f"Hybrid row does not match preregistered config: {mismatched}")


def evaluate_prediction(
    track_b_fusion_benefit: float, prior_benefits: list[float]
) -> dict[str, Any]:
    if len(prior_benefits) != 9:
        raise ValueError("The preregistered reference fit must contain nine cells.")
    prior_max = max(prior_benefits)
    return {
        "status": (
            "confirmed" if track_b_fusion_benefit > prior_max else "not_confirmed"
        ),
        "ranked_largest_fusion_benefit": track_b_fusion_benefit > prior_max,
        "track_a_observation_count": len(prior_benefits),
        "track_a_largest_hybrid_minus_dense": round(prior_max, 6),
    }


def read_latest_by_type(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["retriever_type"]: row for row in csv.DictReader(file)}


def read_prior_fusion_benefits(path: Path) -> list[float]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return [
            float(row["fusion_benefit_hybrid_minus_dense"])
            for row in csv.DictReader(file)
        ]


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
