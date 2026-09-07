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

from rag_bench.compare import compare_metric_values
from rag_bench.generation import ABSTENTION_INSTRUCTION_ON
from scripts.adjust_comparisons import adjust_comparisons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Track B abstention instruction off versus on."
    )
    parser.add_argument("--off-run-name", required=True)
    parser.add_argument("--on-run-name", required=True)
    parser.add_argument("--runs", default="results/track_b_generation_runs.csv")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="results/track_b_abstention_comparisons.csv")
    parser.add_argument(
        "--holm-output", default="results/track_b_abstention_comparisons_holm.csv"
    )
    args = parser.parse_args()

    summaries = read_summaries(Path(args.runs))
    off = summaries[args.off_run_name]
    on = summaries[args.on_run_name]
    validate_abstention_controls(off, on)
    validate_prompt_difference(off, on)
    off_rows = read_per_query(Path(off["per_query_metrics_path"]))
    on_rows = read_per_query(Path(on["per_query_metrics_path"]))

    scopes = {
        "null_abstention_rate": lambda row: row["question_type"] == "null_query",
        "answerable_false_abstention_rate": (
            lambda row: row["question_type"] != "null_query"
        ),
    }
    comparisons: list[dict[str, Any]] = []
    for metric, include in scopes.items():
        baseline = {
            query_id: float(row["abstained"].lower() == "true")
            for query_id, row in off_rows.items()
            if include(row)
        }
        candidate = {
            query_id: float(row["abstained"].lower() == "true")
            for query_id, row in on_rows.items()
            if include(row)
        }
        comparisons.append(
            {
                "comparison": "abstention_instruction_off_to_on",
                "baseline_run_name": off["run_name"],
                "candidate_run_name": on["run_name"],
                "baseline_run_id": off["run_key"],
                "candidate_run_id": on["run_key"],
                **compare_metric_values(
                    baseline,
                    candidate,
                    metric=metric,
                    samples=args.samples,
                    seed=args.seed,
                ),
            }
        )
    write_rows(Path(args.output), comparisons)
    adjust_comparisons(Path(args.output), Path(args.holm_output))


def validate_abstention_controls(off: dict[str, str], on: dict[str, str]) -> None:
    if off.get("abstention_instruction") != "off":
        raise ValueError("The baseline run must have abstention_instruction=off.")
    if on.get("abstention_instruction") != "on":
        raise ValueError("The candidate run must have abstention_instruction=on.")
    allowed_changes = {
        "run_name",
        "run_key",
        "prompt_sha256",
        "abstention_instruction",
        "timestamp_utc",
        "artifact_path",
        "per_query_metrics_path",
        "metadata_path",
        "wall_seconds_this_invocation",
    }
    control_fields = {
        key
        for key in set(off) | set(on)
        if key not in allowed_changes
        and not key.startswith("exact_match")
        and not key.startswith("token_f1")
        and key
        not in {
            "null_abstention_rate",
            "answerable_false_abstention_rate",
            "abstention_separation",
            "response_format_valid_rate",
            "ollama_total_duration_seconds",
            "ollama_load_duration_seconds",
            "ollama_prompt_eval_duration_seconds",
            "ollama_eval_duration_seconds",
            "ollama_duration_component_valid_rate",
            "ollama_duration_component_invalid_count",
            "prompt_eval_tokens_mean",
            "generated_tokens_mean",
            "output_cap_hit_rate",
            "claim_words_p95",
            "claim_words_max",
            "query_latency_p50_ms",
            "query_latency_p95_ms",
        }
    }
    changed = sorted(key for key in control_fields if off.get(key) != on.get(key))
    if changed:
        raise ValueError(f"Only the abstention prompt may change; changed={changed}")


def validate_prompt_difference(off: dict[str, str], on: dict[str, str]) -> None:
    off_metadata = json.loads(Path(off["metadata_path"]).read_text(encoding="utf-8"))
    on_metadata = json.loads(Path(on["metadata_path"]).read_text(encoding="utf-8"))
    if on_metadata["prompt_template"].replace(ABSTENTION_INSTRUCTION_ON, "") != off_metadata[
        "prompt_template"
    ]:
        raise ValueError("Prompt templates differ by more than the abstention instruction.")


def read_summaries(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["run_name"]: row for row in csv.DictReader(file)}


def read_per_query(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["query_id"]: row for row in csv.DictReader(file)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
