from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_bench.compare import compare_metric_values
from rag_bench.multihop import load_multihop_dataset, normalize_space
from scripts.adjust_comparisons import adjust_comparisons
from scripts.compare_track_b_generation import QUESTION_STRATA


DEPTHS = (3, 5, 10, 20)
CONTEXT_BUDGETS = (768, 1280, 2560, 5120)
GENERATION_METRICS = ("exact_match", "token_f1")
MECHANISM_METRICS = (
    "evidence_recall_at_budget",
    "context_sufficiency_at_budget",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the preregistered Track B BM25 context-depth sweep."
    )
    parser.add_argument("--runs", default="results/track_b_generation_runs.csv")
    parser.add_argument("--data-dir", default="data/multihop_rag")
    parser.add_argument(
        "--ordered-run-names",
        required=True,
        help="Comma-separated generation run names in k=3,5,10,20 order.",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--summary-output", default="results/track_b_depth_runs.csv")
    parser.add_argument(
        "--generation-output",
        default="results/track_b_depth_generation_comparisons.csv",
    )
    parser.add_argument(
        "--generation-holm-output",
        default="results/track_b_depth_generation_comparisons_holm.csv",
    )
    parser.add_argument(
        "--mechanism-output",
        default="results/track_b_depth_mechanism_comparisons.csv",
    )
    parser.add_argument(
        "--mechanism-holm-output",
        default="results/track_b_depth_mechanism_comparisons_holm.csv",
    )
    args = parser.parse_args()

    run_names = [item.strip() for item in args.ordered_run_names.split(",") if item.strip()]
    if len(run_names) != len(DEPTHS):
        parser.error("--ordered-run-names must contain runs in k=3,5,10,20 order.")
    summaries = read_summary_rows(Path(args.runs))
    missing = [name for name in run_names if name not in summaries]
    if missing:
        raise ValueError(f"Generation run names are absent from {args.runs}: {missing}")
    selected = [summaries[name] for name in run_names]
    validate_depth_controls(selected)

    dataset = load_multihop_dataset(args.data_dir)
    per_run = [
        evaluate_depth_artifact(Path(row["artifact_path"]), dataset)
        for row in selected
    ]
    summary_rows = build_summary_rows(selected, per_run, dataset)
    generation_rows = build_comparison_rows(
        selected, per_run, dataset, GENERATION_METRICS, args.samples, args.seed
    )
    mechanism_rows = build_comparison_rows(
        selected, per_run, dataset, MECHANISM_METRICS, args.samples, args.seed
    )

    write_rows(Path(args.summary_output), summary_rows)
    write_rows(Path(args.generation_output), generation_rows)
    write_rows(Path(args.mechanism_output), mechanism_rows)
    adjust_comparisons(
        Path(args.generation_output), Path(args.generation_holm_output)
    )
    adjust_comparisons(Path(args.mechanism_output), Path(args.mechanism_holm_output))


def validate_depth_controls(rows: list[dict[str, str]]) -> None:
    if len(rows) != len(DEPTHS):
        raise ValueError(f"G1 requires exactly {len(DEPTHS)} runs.")
    if any(row.get("mode") != "retrieved" for row in rows):
        raise ValueError("G1 requires retrieved-context runs.")
    actual_depths = tuple(int(float(row.get("retrieved_top_k", "0"))) for row in rows)
    actual_budgets = tuple(
        int(float(row.get("max_context_words", "0"))) for row in rows
    )
    if actual_depths != DEPTHS:
        raise ValueError(f"G1 run order must be k={DEPTHS}; found={actual_depths}")
    if actual_budgets != CONTEXT_BUDGETS:
        raise ValueError(
            f"G1 context budgets must be {CONTEXT_BUDGETS}; found={actual_budgets}"
        )
    if any(row.get("evaluation_sample_size") != "300" for row in rows):
        raise ValueError("G1 requires the same 300-query evaluation sample.")
    if any(row.get("evaluation_sample_seed") != "13" for row in rows):
        raise ValueError("G1 requires evaluation sample seed 13.")
    if any(row.get("context_packing") != "balanced" for row in rows):
        raise ValueError("G1 requires balanced context packing.")
    if any(
        row.get("retriever_run_name") != "multihop_bm25_stem_w256_o64"
        for row in rows
    ):
        raise ValueError("G1 requires the frozen BM25 retriever.")

    fixed_fields = (
        "retriever_run_id",
        "retriever_run_name",
        "retriever_config_sha256",
        "retriever_artifact_sha256",
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
        raise ValueError(f"Only depth and context budget may vary in G1; changed={changed}")


def evaluate_depth_artifact(
    path: Path, dataset
) -> dict[str, dict[str, float]]:
    values = {
        metric: {} for metric in (*GENERATION_METRICS, *MECHANISM_METRICS)
    }
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            query_id = str(row["query_id"])
            if dataset.question_types[query_id] == "null_query":
                continue
            annotations = dataset.evidence[query_id]
            context = normalize_space(str(row["context"]))
            covered = sum(
                normalize_space(annotation.fact) in context
                for annotation in annotations
            )
            values["exact_match"][query_id] = float(row["exact_match"])
            values["token_f1"][query_id] = float(row["token_f1"])
            values["evidence_recall_at_budget"][query_id] = covered / len(annotations)
            values["context_sufficiency_at_budget"][query_id] = float(
                covered == len(annotations)
            )
    return values


def values_for_stratum(
    values: dict[str, float], dataset, question_stratum: str
) -> dict[str, float]:
    if question_stratum not in QUESTION_STRATA:
        raise ValueError(f"Unknown question stratum: {question_stratum}")
    return {
        query_id: value
        for query_id, value in values.items()
        if question_stratum == "all_answerable"
        or dataset.question_types[query_id] == question_stratum
    }


def build_summary_rows(
    rows: list[dict[str, str]],
    per_run: list[dict[str, dict[str, float]]],
    dataset,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, values in zip(rows, per_run, strict=True):
        for stratum in QUESTION_STRATA:
            stratum_values = {
                metric: values_for_stratum(metric_values, dataset, stratum)
                for metric, metric_values in values.items()
            }
            output.append(
                {
                    "generation_run_name": row["run_name"],
                    "generation_run_id": row["run_key"],
                    "retrieved_top_k": row["retrieved_top_k"],
                    "max_context_words": row["max_context_words"],
                    "question_stratum": stratum,
                    "num_answerable_queries": len(stratum_values["exact_match"]),
                    **{
                        metric: mean(metric_values.values())
                        for metric, metric_values in stratum_values.items()
                    },
                    "context_words_mean": row.get("context_words_mean", ""),
                    "context_words_p50": row.get("context_words_p50", ""),
                    "context_words_p95": row.get("context_words_p95", ""),
                    "context_words_max": row.get("context_words_max", ""),
                    "prompt_eval_tokens_mean": row.get("prompt_eval_tokens_mean", ""),
                    "query_latency_p50_ms": row.get("query_latency_p50_ms", ""),
                    "query_latency_p95_ms": row.get("query_latency_p95_ms", ""),
                }
            )
    return output


def build_comparison_rows(
    rows: list[dict[str, str]],
    per_run: list[dict[str, dict[str, float]]],
    dataset,
    metrics: tuple[str, ...],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    baseline_index = DEPTHS.index(5)
    for candidate_index, candidate_depth in enumerate(DEPTHS):
        if candidate_index == baseline_index:
            continue
        for stratum in QUESTION_STRATA:
            for metric in metrics:
                baseline_values = values_for_stratum(
                    per_run[baseline_index][metric], dataset, stratum
                )
                candidate_values = values_for_stratum(
                    per_run[candidate_index][metric], dataset, stratum
                )
                result = compare_metric_values(
                    baseline_values,
                    candidate_values,
                    metric=metric,
                    samples=samples,
                    seed=seed,
                )
                output.append(
                    {
                        "question_stratum": stratum,
                        "comparison": f"k5_to_k{candidate_depth}",
                        "baseline_run_name": rows[baseline_index]["run_name"],
                        "candidate_run_name": rows[candidate_index]["run_name"],
                        "baseline_run_id": rows[baseline_index]["run_key"],
                        "candidate_run_id": rows[candidate_index]["run_key"],
                        "baseline_path": rows[baseline_index]["artifact_path"],
                        "candidate_path": rows[candidate_index]["artifact_path"],
                        **result,
                    }
                )
    return output


def read_summary_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["run_name"]: row for row in csv.DictReader(file)}


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
