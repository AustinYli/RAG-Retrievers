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
from scripts.compare_track_b_generation import read_answerable_metric


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare first, middle, and last evidence positions in Track B."
    )
    parser.add_argument(
        "--ordered-run-names",
        required=True,
        help="Comma-separated first,middle,last generation run names.",
    )
    parser.add_argument("--runs", default="results/track_b_generation_runs.csv")
    parser.add_argument("--metrics", default="exact_match,token_f1")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="results/track_b_position_comparisons.csv")
    parser.add_argument(
        "--holm-output", default="results/track_b_position_comparisons_holm.csv"
    )
    args = parser.parse_args()

    run_names = [name.strip() for name in args.ordered_run_names.split(",") if name.strip()]
    if len(run_names) != 3:
        parser.error("--ordered-run-names must contain exactly first,middle,last.")
    summaries = read_summaries(Path(args.runs))
    selected = [summaries[name] for name in run_names]
    validate_position_controls(selected)
    pairs = [
        ("first_to_middle", selected[0], selected[1]),
        ("first_to_last", selected[0], selected[2]),
        ("middle_to_last", selected[1], selected[2]),
    ]

    rows: list[dict[str, Any]] = []
    for metric in [name.strip() for name in args.metrics.split(",") if name.strip()]:
        for comparison, baseline, candidate in pairs:
            result = compare_metric_values(
                read_answerable_metric(Path(baseline["per_query_metrics_path"]), metric),
                read_answerable_metric(Path(candidate["per_query_metrics_path"]), metric),
                metric=metric,
                samples=args.samples,
                seed=args.seed,
            )
            rows.append(
                {
                    "comparison": comparison,
                    "baseline_run_name": baseline["run_name"],
                    "candidate_run_name": candidate["run_name"],
                    "baseline_run_id": baseline["run_key"],
                    "candidate_run_id": candidate["run_key"],
                    **result,
                }
            )
    write_rows(Path(args.output), rows)
    adjust_comparisons(Path(args.output), Path(args.holm_output))


def validate_position_controls(rows: list[dict[str, str]]) -> None:
    positions = [row.get("evidence_position") for row in rows]
    if positions != ["first", "middle", "last"]:
        raise ValueError(
            "Position runs must be ordered first,middle,last; "
            f"found {positions}."
        )
    if any(row.get("mode") != "retrieved" for row in rows):
        raise ValueError("Position comparisons require retrieved-context runs.")
    fixed_fields = (
        "mode",
        "retriever_run_id",
        "retriever_run_name",
        "retriever_config_sha256",
        "retriever_artifact_sha256",
        "retrieved_top_k",
        "max_context_words",
        "context_packing",
        "model",
        "model_digest",
        "model_family",
        "model_parameter_size",
        "quantization_level",
        "ollama_version",
        "temperature",
        "seed",
        "context_window",
        "max_new_tokens",
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
    changed = [
        field for field in fixed_fields if len({row.get(field, "") for row in rows}) != 1
    ]
    if changed:
        raise ValueError(f"Only evidence position may vary; changed={changed}")


def read_summaries(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["run_name"]: row for row in csv.DictReader(file)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
