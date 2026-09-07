from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_bench.generation import PROMPT_DEVELOPMENT_QUERY_COUNT
from rag_bench.multihop import load_multihop_dataset, normalize_space


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribute answer failures using the context actually sent to the generator."
    )
    parser.add_argument("--generation-artifact", required=True)
    parser.add_argument("--data-dir", default="data/multihop_rag")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset = load_multihop_dataset(args.data_dir)
    generated = read_jsonl(Path(args.generation_artifact))
    answerable = {
        query_id: row
        for query_id, row in generated.items()
        if row["question_type"] != "null_query"
    }
    all_answerable_ids = set(dataset.answerable_query_ids)
    evaluation_ids = all_answerable_ids - set(
        dataset.answerable_query_ids[:PROMPT_DEVELOPMENT_QUERY_COUNT]
    )
    valid_complete_sets = {frozenset(all_answerable_ids), frozenset(evaluation_ids)}
    if not args.allow_partial and frozenset(answerable) not in valid_complete_sets:
        missing = sorted(evaluation_ids - set(answerable))
        unexpected = sorted(set(answerable) - all_answerable_ids)
        raise ValueError(
            "Generation artifact does not cover a complete evaluation partition: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    result = attribute_failures(dataset, answerable)
    result["generation_artifact"] = args.generation_artifact
    result["prompt_development_queries_excluded"] = len(
        all_answerable_ids - set(answerable)
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def attribute_failures(dataset, answerable: dict[str, dict]) -> dict:
    buckets = {
        "correct": 0,
        "wrong_missing_complete_gold_evidence": 0,
        "wrong_despite_complete_gold_evidence": 0,
    }
    correct_without_sufficient_context = 0
    sufficient_context_count = 0
    for query_id, row in answerable.items():
        context = normalize_space(str(row["context"]))
        sufficient = all(
            normalize_space(annotation.fact) in context
            for annotation in dataset.evidence[query_id]
        )
        correct = float(row["exact_match"]) == 1.0
        sufficient_context_count += int(sufficient)
        if correct:
            buckets["correct"] += 1
            correct_without_sufficient_context += int(not sufficient)
        elif sufficient:
            buckets["wrong_despite_complete_gold_evidence"] += 1
        else:
            buckets["wrong_missing_complete_gold_evidence"] += 1

    total = len(answerable)
    if total == 0:
        raise ValueError("Generation artifact contains no answerable queries.")
    result = {
        "num_answerable_queries": total,
        "actual_context_sufficiency_count": sufficient_context_count,
        "actual_context_sufficiency_rate": sufficient_context_count / total,
        "correct_without_sufficient_context": correct_without_sufficient_context,
        **buckets,
        **{f"{name}_rate": count / total for name, count in buckets.items()},
    }
    return result


def read_jsonl(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as file:
        return {row["query_id"]: row for row in map(json.loads, file)}


if __name__ == "__main__":
    main()
