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

from rag_bench.multihop import chunk_corpus, load_multihop_dataset
from scripts.run_track_b_stage1 import read_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check Track B Hits@10 against MultiHop-RAG's evaluator convention."
    )
    parser.add_argument("--runs", default="results/track_b_stage1.csv")
    parser.add_argument("--data-dir", default="data/multihop_rag")
    parser.add_argument(
        "--output", default="results/track_b_evidence_metric_crosscheck.json"
    )
    args = parser.parse_args()

    dataset = load_multihop_dataset(args.data_dir)
    with Path(args.runs).open("r", newline="", encoding="utf-8") as file:
        run_rows = list(csv.DictReader(file))

    results: list[dict[str, Any]] = []
    for row in run_rows:
        chunks, _ = chunk_corpus(
            dataset.corpus,
            chunk_words=int(row["chunk_words"]),
            overlap_words=int(row["overlap_words"]),
        )
        run = read_run(Path(row["run_artifact_path"]))
        official_value = mean(
            official_style_hit_at_10(
                run[query_id],
                chunks,
                [item.fact for item in dataset.evidence[query_id]],
            )
            for query_id in dataset.answerable_query_ids
        )
        harness_value = float(row["evidence_hit@10"])
        result = {
            "run_id": row["run_id"],
            "run_name": row["run_name"],
            "official_style_hit@10": round(official_value, 6),
            "harness_evidence_hit@10": round(harness_value, 6),
            "absolute_difference": abs(official_value - harness_value),
        }
        if result["absolute_difference"] > 5e-7:
            raise ValueError(f"Evidence metric cross-check failed: {result}")
        results.append(result)

    payload = {
        "reference_evaluator": (
            "https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py"
        ),
        "checked_metric": "Hits@10 / evidence_hit@10",
        "runs": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def official_style_hit_at_10(
    run: dict[str, float],
    chunks: dict[str, dict[str, Any]],
    gold_facts: list[str],
) -> float:
    gold = [_remove_spaces(fact) for fact in gold_facts]
    ranked = sorted(run.items(), key=lambda item: (-item[1], item[0]))[:10]
    retrieved = [_remove_spaces(str(chunks[chunk_id]["text"])) for chunk_id, _ in ranked]
    return float(any(fact in text for text in retrieved for fact in gold))


def _remove_spaces(text: str) -> str:
    return text.replace(" ", "").replace("\n", "")


if __name__ == "__main__":
    main()
