from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from rag_bench.load import load_beir_dataset
from rag_bench.metrics import evaluate, evaluate_per_query
from rag_bench.rerank import rerank_with_latencies
from rag_bench.run import append_result, percentile


RERANK_TOP_K_VALUES = [10, 20, 50, 100, 200]
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep rerank_top_k over the best base per dataset.")
    parser.add_argument("--runs", default="results/track_a_runs.csv")
    parser.add_argument("--output", default="results/rerank_sweeps.csv")
    parser.add_argument("--per-query-dir", default="results/rerank_sweep_per_query")
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--datasets", help="Comma-separated dataset filter.")
    parser.add_argument("--base-run-names", help="Comma-separated base run-name filter.")
    parser.add_argument("--rerank-top-k-values", help="Comma-separated rerank depths. Defaults to 10,20,50,100,200.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace the output CSV before writing rows.")
    args = parser.parse_args()

    rows = latest_rows_by_run_name(Path(args.runs))
    output_path = Path(args.output)
    per_query_dir = Path(args.per_query_dir)
    if args.force and output_path.exists():
        output_path.unlink()
    rerank_top_k_values = parse_int_csv(args.rerank_top_k_values) or RERANK_TOP_K_VALUES
    model_started = time.perf_counter()
    model = CrossEncoder(
        args.reranker_model,
        max_length=args.max_seq_length,
        trust_remote_code=args.trust_remote_code,
    )
    model_load_seconds = time.perf_counter() - model_started

    for dataset, base_rows in selected_bases_by_dataset(
        rows,
        datasets=parse_str_csv(args.datasets),
        base_run_names=parse_str_csv(args.base_run_names),
        limit=2,
    ).items():
        corpus, queries, qrels = load_beir_dataset(dataset=dataset, split="test", data_dir="data/beir")
        for base_row in base_rows:
            base_run = read_run(Path(base_row["run_artifact_path"]))
            base_latency_p50 = parse_float(base_row.get("query_latency_p50_ms"))
            base_latency_p95 = parse_float(base_row.get("query_latency_p95_ms"))
            for rerank_top_k in rerank_top_k_values:
                started = time.perf_counter()
                run, rerank_latencies = rerank_with_latencies(
                    queries=queries,
                    corpus=corpus,
                    base_runs=base_run,
                    model=model,
                    rerank_top_k=rerank_top_k,
                    keep_top_k=1000,
                    batch_size=args.batch_size,
                )
                search_seconds = time.perf_counter() - started
                metrics = evaluate(qrels, run, [1, 5, 10, 100])
                per_query = evaluate_per_query(qrels, run, [1, 5, 10, 100])
                run_name = (
                    f"{dataset}_rerank_sweep_{model_slug(args.reranker_model)}_"
                    f"{base_row['run_name']}_top{rerank_top_k}"
                )
                per_query_path = write_per_query_metrics(per_query_dir, run_name, per_query)
                row: dict[str, Any] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "dataset": dataset,
                    "base_run_id": base_row["run_id"],
                    "base_run_name": base_row["run_name"],
                    "base_retriever_type": base_row["retriever_type"],
                    "base_ndcg@10": base_row.get("ndcg@10", ""),
                    "dense_model": base_row.get("dense_model", ""),
                    "reranker_model": args.reranker_model,
                    "rerank_top_k": rerank_top_k,
                    "top_k": 1000,
                    "max_seq_length": args.max_seq_length,
                    "batch_size": args.batch_size,
                    "model_load_seconds": round(model_load_seconds, 3),
                    "search_seconds": round(search_seconds, 3),
                    "throughput_ms_per_query": round(search_seconds / len(queries) * 1000.0, 3),
                    "rerank_latency_p50_ms": round(percentile(rerank_latencies, 50) * 1000.0, 3),
                    "rerank_latency_p95_ms": round(percentile(rerank_latencies, 95) * 1000.0, 3),
                    "query_latency_p50_ms": round(base_latency_p50 + percentile(rerank_latencies, 50) * 1000.0, 3),
                    "query_latency_p95_ms": round(base_latency_p95 + percentile(rerank_latencies, 95) * 1000.0, 3),
                    "per_query_metrics_path": str(per_query_path),
                }
                row.update({name: round(value, 6) for name, value in metrics.items()})
                append_result(output_path, row)
                print(json.dumps(row, indent=2, sort_keys=True))


def latest_rows_by_run_name(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows[row["run_name"]] = row
    return rows


def best_bases_by_dataset(rows: dict[str, dict[str, str]], limit: int) -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = {}
    for row in rows.values():
        if row.get("retriever_type") not in {"dense", "hybrid"}:
            continue
        dataset = row["dataset"]
        candidates.setdefault(dataset, []).append(row)
    return {
        dataset: sorted(dataset_rows, key=lambda row: float(row["ndcg@10"]), reverse=True)[:limit]
        for dataset, dataset_rows in candidates.items()
    }


def selected_bases_by_dataset(
    rows: dict[str, dict[str, str]],
    datasets: list[str],
    base_run_names: list[str],
    limit: int,
) -> dict[str, list[dict[str, str]]]:
    if base_run_names:
        selected: dict[str, list[dict[str, str]]] = {}
        for run_name in base_run_names:
            row = rows[run_name]
            if datasets and row["dataset"] not in datasets:
                continue
            selected.setdefault(row["dataset"], []).append(row)
        return selected

    selected = best_bases_by_dataset(rows, limit=limit)
    if not datasets:
        return selected
    return {dataset: base_rows for dataset, base_rows in selected.items() if dataset in datasets}


def parse_float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            run.setdefault(item["query_id"], {})[item["doc_id"]] = float(item["score"])
    return run


def parse_int_csv(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def model_slug(model_name: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in model_name.lower()).strip("-")


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
