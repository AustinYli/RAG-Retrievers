from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_bench.compare import compare_metric_values
from scripts.adjust_comparisons import adjust_comparisons


QUESTION_STRATA = (
    "all_answerable",
    "comparison_query",
    "inference_query",
    "temporal_query",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three Track B generation runs with paired bootstrap intervals."
    )
    parser.add_argument("--runs", default="results/track_b_generation_runs.csv")
    parser.add_argument(
        "--ordered-run-names",
        required=True,
        help="Comma-separated BM25,dense,reranker generation run names.",
    )
    parser.add_argument("--metrics", default="exact_match,token_f1")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="results/track_b_generation_comparisons.csv")
    parser.add_argument(
        "--holm-output", default="results/track_b_generation_comparisons_holm.csv"
    )
    args = parser.parse_args()

    run_names = [item.strip() for item in args.ordered_run_names.split(",") if item.strip()]
    if len(run_names) != 3:
        parser.error("--ordered-run-names must contain exactly BM25,dense,reranker.")
    summaries = read_summary_rows(Path(args.runs))
    missing = [name for name in run_names if name not in summaries]
    if missing:
        raise ValueError(f"Generation run names are absent from {args.runs}: {missing}")
    selected = [summaries[name] for name in run_names]
    validate_generation_controls(selected)
    pairs = [
        ("bm25_to_dense", selected[0], selected[1]),
        ("dense_to_reranker", selected[1], selected[2]),
        ("bm25_to_reranker", selected[0], selected[2]),
    ]

    rows: list[dict[str, Any]] = []
    for question_stratum in QUESTION_STRATA:
        for metric in [item.strip() for item in args.metrics.split(",") if item.strip()]:
            for comparison_name, baseline, candidate in pairs:
                baseline_values = read_answerable_metric(
                    Path(baseline["per_query_metrics_path"]),
                    metric,
                    question_stratum=question_stratum,
                )
                candidate_values = read_answerable_metric(
                    Path(candidate["per_query_metrics_path"]),
                    metric,
                    question_stratum=question_stratum,
                )
                result = compare_metric_values(
                    baseline_values,
                    candidate_values,
                    metric=metric,
                    samples=args.samples,
                    seed=args.seed,
                )
                rows.append(
                    {
                        "question_stratum": question_stratum,
                        "comparison": comparison_name,
                        "baseline_run_name": baseline["run_name"],
                        "candidate_run_name": candidate["run_name"],
                        "baseline_run_id": baseline["run_key"],
                        "candidate_run_id": candidate["run_key"],
                        "baseline_path": baseline["per_query_metrics_path"],
                        "candidate_path": candidate["per_query_metrics_path"],
                        **result,
                    }
                )

    write_rows(Path(args.output), rows)
    adjust_comparisons(Path(args.output), Path(args.holm_output))


def read_summary_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = {row["run_name"]: row for row in csv.DictReader(file)}
    return rows


def validate_generation_controls(rows: list[dict[str, str]]) -> None:
    fixed_fields = (
        "mode",
        "model",
        "model_digest",
        "quantization_level",
        "ollama_version",
        "temperature",
        "seed",
        "context_window",
        "max_new_tokens",
        "retrieved_top_k",
        "max_context_words",
        "context_packing",
        "evidence_position",
        "abstention_instruction",
        "prompt_sha256",
        "answer_normalizer_version",
        "response_parser_version",
        "context_builder_version",
        "evaluation_partition_version",
        "prompt_development_query_count",
        "prompt_development_query_ids_sha256",
        "abstention_patterns_sha256",
        "repeat",
        "query_scope",
        "selected_query_ids_sha256",
        "evaluation_sample_size",
        "evaluation_sample_seed",
        "evaluation_sampler_version",
        "queries_sha256",
        "corpus_sha256",
        "platform",
        "python_version",
    )
    if any(row.get("mode") != "retrieved" for row in rows):
        raise ValueError("E1 comparisons require three retrieved-context runs.")
    changed = [
        field
        for field in fixed_fields
        if len({row.get(field, "") for row in rows}) != 1
    ]
    if changed:
        raise ValueError(
            "Only the retriever may vary in an E1 comparison; "
            f"changed controls={changed}"
        )


def read_answerable_metric(
    path: Path, metric: str, question_stratum: str = "all_answerable"
) -> dict[str, float]:
    if question_stratum not in QUESTION_STRATA:
        raise ValueError(f"Unknown question stratum: {question_stratum}")
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(f"{path} does not contain metric column {metric!r}.")
        return {
            row["query_id"]: float(row[metric])
            for row in reader
            if row["question_type"] != "null_query"
            and (
                question_stratum == "all_answerable"
                or row["question_type"] == question_stratum
            )
        }


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
