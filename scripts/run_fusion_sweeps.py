from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_bench.load import load_beir_dataset
from rag_bench.metrics import evaluate, evaluate_per_query
from rag_bench.retrievers import hybrid_rrf
from rag_bench.run import append_result, depth_stats


RRF_K_VALUES = [0, 1, 2, 5, 10, 20, 60, 100, 200, 1000]
WEIGHT_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0]
ASYMMETRIC_DEPTHS = [(100, 100), (100, 1000), (1000, 100), (1000, 1000)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cheap fusion sweeps from saved BM25/dense artifacts.")
    parser.add_argument("--runs", default="results/track_a_runs.csv")
    parser.add_argument("--output", default="results/fusion_sweeps.csv")
    parser.add_argument("--per-query-dir", default="results/fusion_sweep_per_query")
    parser.add_argument("--force", action="store_true", help="Replace the output CSV before writing rows.")
    args = parser.parse_args()

    runs = latest_rows_by_run_name(Path(args.runs))
    output_path = Path(args.output)
    per_query_dir = Path(args.per_query_dir)
    if args.force and output_path.exists():
        output_path.unlink()
    best_dense = best_dense_by_dataset(runs)
    for dataset, dense_row in best_dense.items():
        bm25_row = runs[f"{dataset}_bm25_stem_512"]
        bm25_run = read_run(Path(bm25_row["run_artifact_path"]))
        dense_run = read_run(Path(dense_row["run_artifact_path"]))
        _, _, qrels = load_beir_dataset(dataset=dataset, split="test", data_dir="data/beir")

        for rrf_k in RRF_K_VALUES:
            row = score_fusion(
                dataset=dataset,
                sweep_name="rrf_k",
                bm25_row=bm25_row,
                dense_row=dense_row,
                bm25_run=bm25_run,
                dense_run=dense_run,
                qrels=qrels,
                rrf_k=rrf_k,
                bm25_weight=1.0,
                dense_weight=1.0,
                bm25_depth=1000,
                dense_depth=1000,
                per_query_dir=per_query_dir,
            )
            append_result(output_path, row)
            print(json.dumps(row, indent=2, sort_keys=True))

        for dense_weight in WEIGHT_VALUES:
            row = score_fusion(
                dataset=dataset,
                sweep_name="dense_weight",
                bm25_row=bm25_row,
                dense_row=dense_row,
                bm25_run=bm25_run,
                dense_run=dense_run,
                qrels=qrels,
                rrf_k=60,
                bm25_weight=1.0,
                dense_weight=dense_weight,
                bm25_depth=1000,
                dense_depth=1000,
                per_query_dir=per_query_dir,
            )
            append_result(output_path, row)
            print(json.dumps(row, indent=2, sort_keys=True))

    dataset = "nfcorpus"
    bm25_row = runs[f"{dataset}_bm25_stem_512"]
    dense_row = best_dense[dataset]
    bm25_run = read_run(Path(bm25_row["run_artifact_path"]))
    dense_run = read_run(Path(dense_row["run_artifact_path"]))
    _, _, qrels = load_beir_dataset(dataset=dataset, split="test", data_dir="data/beir")
    for bm25_depth, dense_depth in ASYMMETRIC_DEPTHS:
        row = score_fusion(
            dataset=dataset,
            sweep_name="asymmetric_depth",
            bm25_row=bm25_row,
            dense_row=dense_row,
            bm25_run=bm25_run,
            dense_run=dense_run,
            qrels=qrels,
            rrf_k=60,
            bm25_weight=1.0,
            dense_weight=1.0,
            bm25_depth=bm25_depth,
            dense_depth=dense_depth,
            per_query_dir=per_query_dir,
        )
        append_result(output_path, row)
        print(json.dumps(row, indent=2, sort_keys=True))


def score_fusion(
    dataset: str,
    sweep_name: str,
    bm25_row: dict[str, str],
    dense_row: dict[str, str],
    bm25_run: dict[str, dict[str, float]],
    dense_run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
    bm25_depth: int,
    dense_depth: int,
    per_query_dir: Path,
) -> dict[str, Any]:
    truncated_bm25 = truncate_run(bm25_run, bm25_depth)
    truncated_dense = truncate_run(dense_run, dense_depth)
    run = hybrid_rrf(
        truncated_bm25,
        truncated_dense,
        top_k=1000,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )
    metrics = evaluate(qrels, run, [1, 5, 10, 100])
    per_query = evaluate_per_query(qrels, run, [1, 5, 10, 100])
    run_name = (
        f"{dataset}_{sweep_name}_{dense_row['run_name']}_rrf{rrf_k}_"
        f"bw{bm25_weight}_dw{dense_weight}_bd{bm25_depth}_dd{dense_depth}"
    )
    per_query_path = write_per_query_metrics(per_query_dir, run_name, per_query)
    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "dataset": dataset,
        "sweep_name": sweep_name,
        "bm25_run_id": bm25_row["run_id"],
        "dense_run_id": dense_row["run_id"],
        "dense_model": dense_row["dense_model"],
        "rrf_k": rrf_k,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "bm25_depth": bm25_depth,
        "dense_depth": dense_depth,
        "bm25_realized_depth_mean": depth_stats(truncated_bm25, "bm25_realized_depth")[
            "bm25_realized_depth_mean"
        ],
        "dense_realized_depth_mean": depth_stats(truncated_dense, "dense_realized_depth")[
            "dense_realized_depth_mean"
        ],
        "realized_depth_mean": depth_stats(run, "realized_depth")["realized_depth_mean"],
        "per_query_metrics_path": str(per_query_path),
    }
    row.update({name: round(value, 6) for name, value in metrics.items()})
    return row


def latest_rows_by_run_name(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows[row["run_name"]] = row
    return rows


def best_dense_by_dataset(rows: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows.values():
        if row.get("retriever_type") != "dense":
            continue
        dataset = row["dataset"]
        if dataset not in best or float(row["ndcg@10"]) > float(best[dataset]["ndcg@10"]):
            best[dataset] = row
    return best


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            run.setdefault(item["query_id"], {})[item["doc_id"]] = float(item["score"])
    return run


def truncate_run(run: dict[str, dict[str, float]], depth: int) -> dict[str, dict[str, float]]:
    truncated: dict[str, dict[str, float]] = {}
    for query_id, scores in run.items():
        truncated[query_id] = dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:depth])
    return truncated


def write_per_query_metrics(
    output_dir: Path,
    run_name: str,
    per_query: dict[str, dict[str, float]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_name}.per_query.csv"
    metric_fields = sorted({metric for metrics in per_query.values() for metric in metrics})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["query_id", *metric_fields], lineterminator="\n")
        writer.writeheader()
        for query_id in sorted(per_query):
            writer.writerow({"query_id": query_id, **per_query[query_id]})
    return path


if __name__ == "__main__":
    main()
