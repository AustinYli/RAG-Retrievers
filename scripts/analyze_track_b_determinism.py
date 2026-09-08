from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure run-to-run generation spread.")
    parser.add_argument("artifacts", nargs=3, help="Three generation JSONL artifacts.")
    parser.add_argument("--output")
    args = parser.parse_args()

    runs = [read_jsonl(Path(path)) for path in args.artifacts]
    metadata_paths = [metadata_path_for(Path(path)) for path in args.artifacts]
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    validate_determinism_metadata(metadata)
    result = summarize_determinism(runs)
    result.update(
        {
            "artifacts": args.artifacts,
            "metadata_artifacts": [str(path) for path in metadata_paths],
        }
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def summarize_determinism(runs: list[dict[str, dict]]) -> dict:
    query_sets = [set(run) for run in runs]
    if not all(query_ids == query_sets[0] for query_ids in query_sets[1:]):
        raise ValueError("Determinism runs must contain identical query IDs.")
    query_ids = sorted(query_sets[0])
    answerable_ids = [
        query_id
        for query_id in query_ids
        if runs[0][query_id]["question_type"] != "null_query"
    ]
    if not answerable_ids:
        raise ValueError("Determinism analysis requires answerable questions for EM/F1.")
    em = [
        mean(float(run[qid]["exact_match"]) for qid in answerable_ids)
        for run in runs
    ]
    f1 = [
        mean(float(run[qid]["token_f1"]) for qid in answerable_ids)
        for run in runs
    ]
    prediction_agreement = mean(
        len({run[qid]["prediction"] for run in runs}) == 1 for qid in query_ids
    )
    return {
        "num_queries": len(query_ids),
        "num_answerable_queries": len(answerable_ids),
        "num_null_queries": len(query_ids) - len(answerable_ids),
        "exact_match_by_run": em,
        "exact_match_mean": mean(em),
        "exact_match_population_stddev": pstdev(em),
        "exact_match_range": max(em) - min(em),
        "token_f1_by_run": f1,
        "token_f1_mean": mean(f1),
        "token_f1_population_stddev": pstdev(f1),
        "token_f1_range": max(f1) - min(f1),
        "exact_prediction_agreement_rate": prediction_agreement,
    }


def read_jsonl(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as file:
        return {row["query_id"]: row for row in map(json.loads, file)}


def metadata_path_for(artifact_path: Path) -> Path:
    if artifact_path.name.endswith(".jsonl"):
        return artifact_path.with_name(
            artifact_path.name[: -len(".jsonl")] + ".metadata.json"
        )
    raise ValueError(f"Expected a JSONL generation artifact: {artifact_path}")


def validate_determinism_metadata(metadata: list[dict]) -> None:
    controls = []
    for item in metadata:
        experiment = dict(item["experiment"])
        experiment.pop("repeat", None)
        controls.append(
            {
                "experiment": experiment,
                "prompt_template": item["prompt_template"],
                "model_details": item["model_details"],
                "runtime": item["runtime"],
            }
        )
    if any(control != controls[0] for control in controls[1:]):
        raise ValueError(
            "Determinism artifacts differ in controls other than the repeat number."
        )


if __name__ == "__main__":
    main()
