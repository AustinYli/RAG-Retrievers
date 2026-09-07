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
from scripts.compare_track_b_generation import validate_generation_controls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare matched-context-budget Track B chunking runs."
    )
    parser.add_argument("--runs", default="results/track_b_generation_runs.csv")
    parser.add_argument("--retrieval-runs", default="results/track_b_stage1.csv")
    parser.add_argument("--data-dir", default="data/multihop_rag")
    parser.add_argument("--baseline-run-name", required=True)
    parser.add_argument(
        "--candidate-run-names",
        required=True,
        help="Three comma-separated generation runs: no-overlap, structural, semantic.",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--summary-output", default="results/track_b_chunking_runs.csv")
    parser.add_argument(
        "--output", default="results/track_b_chunking_comparisons.csv"
    )
    parser.add_argument(
        "--holm-output", default="results/track_b_chunking_comparisons_holm.csv"
    )
    args = parser.parse_args()

    candidate_names = [
        item.strip() for item in args.candidate_run_names.split(",") if item.strip()
    ]
    if len(candidate_names) != 3:
        parser.error("--candidate-run-names must contain exactly three runs.")
    generation_rows = read_rows(Path(args.runs))
    ordered_names = [args.baseline_run_name, *candidate_names]
    missing = [name for name in ordered_names if name not in generation_rows]
    if missing:
        raise ValueError(f"Generation runs are missing from {args.runs}: {missing}")
    selected = [generation_rows[name] for name in ordered_names]
    validate_generation_controls(selected)
    if any(row.get("context_packing") != "greedy" for row in selected):
        raise ValueError("Chunking comparisons require context_packing=greedy.")

    retrieval_rows = read_rows(Path(args.retrieval_runs))
    selected_retrieval = []
    for generation_row in selected:
        retriever_name = generation_row["retriever_run_name"]
        if retriever_name not in retrieval_rows:
            raise ValueError(f"Retrieval run is missing: {retriever_name}")
        selected_retrieval.append(retrieval_rows[retriever_name])
    validate_retrieval_controls(selected_retrieval)

    dataset = load_multihop_dataset(args.data_dir)
    per_run: list[dict[str, dict[str, float]]] = []
    summary_rows: list[dict[str, Any]] = []
    for generation_row, retrieval_row in zip(
        selected, selected_retrieval, strict=True
    ):
        values, context_counts = evaluate_generation_artifact(
            Path(generation_row["artifact_path"]), dataset
        )
        per_run.append(values)
        summary_rows.append(
            {
                "generation_run_name": generation_row["run_name"],
                "generation_run_key": generation_row["run_key"],
                "retriever_run_name": retrieval_row["run_name"],
                "retriever_run_id": retrieval_row["run_id"],
                "chunking_label": chunking_label(retrieval_row),
                "chunking_strategy": retrieval_row.get("chunking_strategy") or "fixed",
                "chunk_words": retrieval_row["chunk_words"],
                "overlap_words": retrieval_row["overlap_words"],
                "corpus_chunks": retrieval_row["corpus_chunks"],
                "chunks_per_document": retrieval_row["chunks_per_document"],
                "max_context_words": generation_row["max_context_words"],
                "context_words_mean": generation_row["context_words_mean"],
                "realized_context_chunks_mean": mean(context_counts),
                "num_answerable_queries": len(values["exact_match"]),
                **{
                    metric: mean(metric_values.values())
                    for metric, metric_values in values.items()
                },
            }
        )

    comparison_rows: list[dict[str, Any]] = []
    baseline_values = per_run[0]
    for candidate_index in range(1, len(selected)):
        for metric in (
            "exact_match",
            "token_f1",
            "evidence_recall_at_budget",
            "context_sufficiency_at_budget",
        ):
            result = compare_metric_values(
                baseline_values[metric],
                per_run[candidate_index][metric],
                metric=metric,
                samples=args.samples,
                seed=args.seed,
            )
            comparison_rows.append(
                {
                    "comparison": (
                        f"{chunking_label(selected_retrieval[0])}_to_"
                        f"{chunking_label(selected_retrieval[candidate_index])}"
                    ),
                    "baseline_run_name": selected[0]["run_name"],
                    "candidate_run_name": selected[candidate_index]["run_name"],
                    "baseline_run_id": selected[0]["run_key"],
                    "candidate_run_id": selected[candidate_index]["run_key"],
                    "baseline_path": selected[0]["artifact_path"],
                    "candidate_path": selected[candidate_index]["artifact_path"],
                    **result,
                }
            )

    write_rows(Path(args.summary_output), summary_rows)
    write_rows(Path(args.output), comparison_rows)
    adjust_comparisons(Path(args.output), Path(args.holm_output))


def evaluate_generation_artifact(path: Path, dataset) -> tuple[
    dict[str, dict[str, float]], list[int]
]:
    values = {
        "exact_match": {},
        "token_f1": {},
        "evidence_recall_at_budget": {},
        "context_sufficiency_at_budget": {},
    }
    context_counts: list[int] = []
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
            values["evidence_recall_at_budget"][query_id] = covered / len(
                annotations
            )
            values["context_sufficiency_at_budget"][query_id] = float(
                covered == len(annotations)
            )
            context_counts.append(len(row["context_ids"]))
    return values, context_counts


def validate_retrieval_controls(rows: list[dict[str, str]]) -> None:
    if any(row.get("retriever_type") != "bm25" for row in rows):
        raise ValueError("The frozen E5 retriever must be BM25 in every run.")
    fixed_fields = (
        "dataset",
        "num_queries_retrieved",
        "num_answerable_queries_scored",
        "num_null_queries",
        "corpus_documents",
        "chunk_words",
        "chunk_unit",
        "corpus_text",
        "retrieval_top_k",
        "bm25_analyzer",
        "bm25_k1",
        "bm25_b",
        "index_backend",
        "queries_sha256",
        "corpus_sha256",
    )
    changed = [
        field for field in fixed_fields if len({row.get(field, "") for row in rows}) != 1
    ]
    if changed:
        raise ValueError(f"Only segmentation may vary in E5; changed={changed}")
    labels = [chunking_label(row) for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Chunking labels must be unique; found={labels}")


def chunking_label(row: dict[str, str]) -> str:
    strategy = row.get("chunking_strategy") or "fixed"
    if strategy == "fixed":
        return f"fixed_overlap_{int(float(row['overlap_words']))}"
    return strategy


def read_rows(path: Path) -> dict[str, dict[str, str]]:
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
