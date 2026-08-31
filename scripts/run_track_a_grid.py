from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rag_bench.run import append_result, run_config


DATASETS = ["scifact", "nfcorpus", "fiqa"]
MODELS = [
    ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    ("bge", "BAAI/bge-base-en-v1.5"),
    ("e5", "intfloat/e5-base-v2"),
]

BM25_CONFIG = {
    "analyzer": "word-lower-stop-porter-v1",
    "k1": 1.2,
    "b": 0.9,
    "epsilon": 0.25,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed Track A BM25/dense/hybrid grid.")
    parser.add_argument("--results-path", default="results/track_a_runs.csv")
    parser.add_argument("--artifacts-dir", default="results/track_a_artifacts")
    parser.add_argument("--stage", choices=["all", "bm25", "dense", "hybrid"], default="all")
    parser.add_argument("--force", action="store_true", help="Run even when run_name already exists in results.")
    args = parser.parse_args()

    results_path = Path(args.results_path)
    seen = set() if args.force else existing_run_names(results_path)
    for config in iter_configs(args.stage, results_path, Path(args.artifacts_dir)):
        if config["run_name"] in seen:
            print(f"SKIP {config['run_name']}")
            continue
        print(f"RUN {config['run_name']}")
        row = run_config(config, Path(f"<generated:{config['run_name']}>"))
        append_result(results_path, row)
        print(json.dumps(row, indent=2, sort_keys=True))


def iter_configs(stage: str, results_path: Path, artifacts_dir: Path):
    if stage in {"all", "bm25"}:
        for dataset in DATASETS:
            yield base_config(
                run_name=f"{dataset}_bm25_stem_512",
                dataset=dataset,
                results_path=results_path,
                artifacts_dir=artifacts_dir,
                retriever={
                    "type": "bm25",
                    "top_k": 1000,
                    "max_seq_length": 512,
                    **BM25_CONFIG,
                },
            )

    if stage in {"all", "dense"}:
        for model_slug, model_name in MODELS:
            for dataset in DATASETS:
                yield base_config(
                    run_name=f"{dataset}_dense_{model_slug}_512",
                    dataset=dataset,
                    results_path=results_path,
                    artifacts_dir=artifacts_dir,
                    retriever={
                        "type": "dense",
                        "dense_model": model_name,
                        "index_backend": "numpy",
                        "top_k": 1000,
                        "batch_size": 64,
                        "max_seq_length": 512,
                        "normalize_embeddings": True,
                        "device": None,
                    },
                )

    if stage in {"all", "hybrid"}:
        for model_slug, model_name in MODELS:
            for dataset in DATASETS:
                yield base_config(
                    run_name=f"{dataset}_hybrid_stem_{model_slug}_512",
                    dataset=dataset,
                    results_path=results_path,
                    artifacts_dir=artifacts_dir,
                    retriever={
                        "type": "hybrid",
                        "dense_model": model_name,
                        "index_backend": "numpy",
                        "top_k": 1000,
                        "bm25_top_k": 1000,
                        "dense_top_k": 1000,
                        "rrf_k": 60,
                        "bm25_weight": 1.0,
                        "dense_weight": 1.0,
                        "batch_size": 64,
                        "max_seq_length": 512,
                        "normalize_embeddings": True,
                        "device": None,
                        **BM25_CONFIG,
                    },
                )


def base_config(
    run_name: str,
    dataset: str,
    results_path: Path,
    artifacts_dir: Path,
    retriever: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "dataset": dataset,
        "split": "test",
        "data_dir": "data/beir",
        "cache_dir": "cache",
        "results_path": str(results_path),
        "artifacts_dir": str(artifacts_dir),
        "retriever": retriever,
        "metrics": {"cutoffs": [1, 5, 10, 100]},
    }


def existing_run_names(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    with results_path.open("r", newline="", encoding="utf-8") as file:
        return {row["run_name"] for row in csv.DictReader(file)}


if __name__ == "__main__":
    main()
