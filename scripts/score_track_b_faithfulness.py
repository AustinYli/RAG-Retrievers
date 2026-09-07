from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean
from typing import Any

from sentence_transformers import CrossEncoder

from rag_bench.faithfulness import (
    claim_support_scores,
    cohen_kappa_binary,
    split_claims,
    unsupported_claim_rate,
)


DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
SCORING_VERSION = "structured-claim-atomic-block-max-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated claims with a local NLI model.")
    parser.add_argument("--generation-artifact", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--calibration-threshold", type=float, default=0.5)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--summary-output")
    parser.add_argument("--calibration-labels")
    parser.add_argument("--minimum-calibration-labels", type=int, default=50)
    parser.add_argument(
        "--include-abstentions",
        action="store_true",
        help="Include refusal claims, whose absence assertions NLI cannot directly verify.",
    )
    args = parser.parse_args()

    if (
        args.max_seq_length <= 0
        or args.batch_size <= 0
        or args.minimum_calibration_labels <= 0
    ):
        parser.error(
            "--max-seq-length, --batch-size, and --minimum-calibration-labels "
            "must be positive."
        )

    generation_path = Path(args.generation_artifact)
    output_path = Path(args.output) if args.output else generation_path.with_suffix(".claims.jsonl")
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else generation_path.with_suffix(".claims.summary.json")
    )
    thresholds = [float(item.strip()) for item in args.thresholds.split(",")]
    if not thresholds or any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("Every threshold must be between zero and one.")
    if not 0.0 <= args.calibration_threshold <= 1.0:
        raise ValueError("Calibration threshold must be between zero and one.")
    model_revision = args.model_revision or (
        DEFAULT_REVISION if args.model == DEFAULT_MODEL else None
    )
    model = CrossEncoder(
        args.model,
        revision=model_revision,
        max_length=args.max_seq_length,
        device=args.device,
    )
    generation_rows = read_jsonl(generation_path)
    claim_rows: list[dict[str, Any]] = []
    unsupported_counts = {threshold: 0 for threshold in thresholds}
    scoped_unsupported_counts = {
        scope: {threshold: 0 for threshold in thresholds}
        for scope in ("answerable", "null_nonabstained")
    }
    scoped_claim_counts = {"answerable": 0, "null_nonabstained": 0}
    answer_rates_by_threshold = {threshold: [] for threshold in thresholds}
    scored_generation_rows = [
        row
        for row in generation_rows
        if args.include_abstentions or not bool(row.get("abstained", False))
    ]
    for row in scored_generation_rows:
        faithfulness_text = str(row.get("faithfulness_text", row["prediction"]))
        structured_claim = bool(row.get("response_format_valid", False))
        claims = [faithfulness_text] if structured_claim else split_claims(faithfulness_text)
        scores = claim_support_scores(
            str(row["context"]),
            faithfulness_text,
            model,
            batch_size=args.batch_size,
            split_sentences=not structured_claim,
        )
        scope = (
            "null_nonabstained"
            if row.get("question_type") == "null_query"
            else "answerable"
        )
        scoped_claim_counts[scope] += len(scores)
        for claim_index, (claim, score) in enumerate(zip(claims, scores, strict=True)):
            claim_rows.append(
                {
                    "query_id": row["query_id"],
                    "claim_index": claim_index,
                    "claim": claim,
                    "question": row.get("question", ""),
                    "question_type": row.get("question_type", ""),
                    "context": row["context"],
                    "entailment_probability": score,
                }
            )
        for threshold in thresholds:
            unsupported = sum(score < threshold for score in scores)
            unsupported_counts[threshold] += unsupported
            scoped_unsupported_counts[scope][threshold] += unsupported
            answer_rates_by_threshold[threshold].append(unsupported_claim_rate(scores, threshold))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in claim_rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "generation_artifact": str(generation_path),
        "claim_artifact": str(output_path),
        "nli_model": args.model,
        "nli_model_revision": model_revision or "unresolved",
        "generation_artifact_sha256": file_sha256(generation_path),
        "faithfulness_scoring_version": SCORING_VERSION,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "device": str(model.device),
        "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
        "num_answers": len(generation_rows),
        "num_answers_scored": len(scored_generation_rows),
        "num_abstentions_excluded": len(generation_rows) - len(scored_generation_rows),
        "include_abstentions": args.include_abstentions,
        "num_claims": len(claim_rows),
        "num_answerable_claims": scoped_claim_counts["answerable"],
        "num_null_nonabstained_claims": scoped_claim_counts["null_nonabstained"],
        **{
            f"unsupported_claim_rate_threshold_{threshold:g}": (
                unsupported_counts[threshold] / len(claim_rows) if claim_rows else 0.0
            )
            for threshold in thresholds
        },
        **{
            f"mean_answer_unsupported_rate_threshold_{threshold:g}": mean(rates)
            for threshold, rates in answer_rates_by_threshold.items()
        },
        **{
            f"{scope}_unsupported_claim_rate_threshold_{threshold:g}": (
                scoped_unsupported_counts[scope][threshold]
                / scoped_claim_counts[scope]
                if scoped_claim_counts[scope]
                else None
            )
            for scope in scoped_claim_counts
            for threshold in thresholds
        },
    }
    if args.calibration_labels:
        summary.update(
            calibration_metrics(
                claim_rows,
                Path(args.calibration_labels),
                threshold=args.calibration_threshold,
                minimum_labels=args.minimum_calibration_labels,
            )
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def calibration_metrics(
    claim_rows: list[dict[str, Any]],
    labels_path: Path,
    threshold: float,
    minimum_labels: int = 50,
) -> dict[str, float]:
    scores = {
        (row["query_id"], int(row["claim_index"])): float(row["entailment_probability"])
        for row in claim_rows
    }
    expected: list[bool] = []
    observed: list[bool] = []
    with labels_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("human_supported") not in {"0", "1"}:
                continue
            key = (row["query_id"], int(row["claim_index"]))
            expected.append(row["human_supported"] == "1")
            observed.append(scores[key] >= threshold)
    if len(expected) < minimum_labels:
        raise ValueError(
            f"Calibration requires at least {minimum_labels} completed labels; "
            f"found {len(expected)}."
        )
    return {
        "calibration_claims": len(expected),
        "calibration_threshold": threshold,
        "calibration_cohen_kappa": cohen_kappa_binary(expected, observed),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
