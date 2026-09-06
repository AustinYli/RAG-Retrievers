from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sentence_transformers import CrossEncoder

from rag_bench.load import load_beir_dataset
from rag_bench.metrics import evaluate, evaluate_per_query
from rag_bench.rerank import rerank_with_latencies
from rag_bench.retrievers import BM25Retriever, DenseRetriever, hybrid_rrf_with_latencies


@dataclass(frozen=True)
class RunResult:
    run: dict[str, dict[str, float]]
    timings: dict[str, float]
    query_latencies: list[float]
    metadata: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BEIR retrieval benchmark config.")
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    row = run_config(config, config_path)
    append_result(Path(config.get("results_path", "results/runs.csv")), row)
    print(json.dumps(row, indent=2, sort_keys=True))


def run_config(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = config["dataset"]
    split = config.get("split", "test")

    load_started = time.perf_counter()
    corpus, queries, qrels = load_beir_dataset(
        dataset=dataset,
        split=split,
        data_dir=config.get("data_dir", "data/beir"),
        max_corpus_docs=config.get("max_corpus_docs"),
        max_queries=config.get("max_queries"),
    )
    load_seconds = time.perf_counter() - load_started

    retriever_config = config["retriever"]
    run_result = build_run(
        retriever_config=retriever_config,
        corpus=corpus,
        queries=queries,
        dataset=dataset,
        cache_dir=config.get("cache_dir", "cache"),
    )

    cutoffs = config.get("metrics", {}).get("cutoffs", [1, 5, 10, 100])
    metrics = evaluate(qrels, run_result.run, cutoffs)
    per_query = evaluate_per_query(qrels, run_result.run, cutoffs)
    total_seconds = time.perf_counter() - started
    run_name = config.get("run_name", config_path.stem)
    run_id = make_run_id(run_name)
    artifacts_dir = Path(config.get("artifacts_dir", "results/artifacts"))
    artifact_paths = write_artifacts(
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        run=run_result.run,
        per_query=per_query,
        config=config,
        row_metadata={
            "dataset": dataset,
            "split": split,
            "run_name": run_name,
            "config_path": str(config_path),
        },
    )

    search_seconds = run_result.timings.get("search_seconds", 0.0)
    throughput_ms_per_query = (search_seconds / len(queries) * 1000.0) if queries else 0.0
    base_config = retriever_config.get("base", {}) if retriever_config["type"] == "rerank" else {}

    row: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "run_name": run_name,
        "config_path": str(config_path),
        "dataset": dataset,
        "split": split,
        "retriever_type": retriever_config["type"],
        "base_retriever_type": base_config.get("type", retriever_config["type"]),
        "dense_model": first_present(retriever_config, base_config, "dense_model"),
        "reranker_model": retriever_config.get("reranker_model", ""),
        "index_backend": first_present(retriever_config, base_config, "index_backend"),
        "top_k": retriever_config.get("top_k", ""),
        "bm25_top_k": first_present(retriever_config, base_config, "bm25_top_k"),
        "dense_top_k": first_present(retriever_config, base_config, "dense_top_k"),
        "rrf_k": first_present(retriever_config, base_config, "rrf_k"),
        "bm25_weight": first_present(retriever_config, base_config, "bm25_weight"),
        "dense_weight": first_present(retriever_config, base_config, "dense_weight"),
        "rerank_top_k": retriever_config.get("rerank_top_k", ""),
        "bm25_analyzer": first_present(retriever_config, base_config, "analyzer"),
        "bm25_k1": first_present(retriever_config, base_config, "k1"),
        "bm25_b": first_present(retriever_config, base_config, "b"),
        "bm25_epsilon": first_present(retriever_config, base_config, "epsilon"),
        "max_seq_length": first_present(retriever_config, base_config, "max_seq_length"),
        "normalize_embeddings": first_present(retriever_config, base_config, "normalize_embeddings"),
        "device": run_result.metadata.get("device", first_present(retriever_config, base_config, "device")),
        "python_version": platform.python_version(),
        "torch_num_threads": run_result.metadata.get("torch_num_threads", ""),
        "torch_num_interop_threads": run_result.metadata.get("torch_num_interop_threads", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "realized_depth_mean": round(run_result.metadata.get("realized_depth_mean", 0.0), 3),
        "realized_depth_p50": round(run_result.metadata.get("realized_depth_p50", 0.0), 3),
        "realized_depth_p95": round(run_result.metadata.get("realized_depth_p95", 0.0), 3),
        "bm25_realized_depth_mean": round(run_result.metadata.get("bm25_realized_depth_mean", 0.0), 3)
        if "bm25_realized_depth_mean" in run_result.metadata
        else "",
        "dense_realized_depth_mean": round(run_result.metadata.get("dense_realized_depth_mean", 0.0), 3)
        if "dense_realized_depth_mean" in run_result.metadata
        else "",
        "num_corpus_docs": len(corpus),
        "num_queries": len(queries),
        "load_seconds": round(load_seconds, 3),
        "model_load_seconds": round(run_result.timings.get("model_load_seconds", 0.0), 3),
        "index_seconds": round(run_result.timings.get("index_seconds", 0.0), 3),
        "search_seconds": round(search_seconds, 3),
        "throughput_ms_per_query": round(throughput_ms_per_query, 3),
        "query_latency_p50_ms": round(percentile(run_result.query_latencies, 50) * 1000, 3) if run_result.query_latencies else "",
        "query_latency_p95_ms": round(percentile(run_result.query_latencies, 95) * 1000, 3) if run_result.query_latencies else "",
        "total_seconds": round(total_seconds, 3),
        "run_artifact_path": artifact_paths["run_artifact_path"],
        "per_query_metrics_path": artifact_paths["per_query_metrics_path"],
        "metadata_path": artifact_paths["metadata_path"],
    }
    row.update({name: round(value, 6) for name, value in metrics.items()})
    return row


def build_run(
    retriever_config: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    queries: dict[str, str],
    dataset: str,
    cache_dir: str,
) -> RunResult:
    retriever_type = retriever_config["type"]
    top_k = int(retriever_config.get("top_k", 1000))

    if retriever_type == "bm25":
        index_started = time.perf_counter()
        retriever = BM25Retriever(
            corpus,
            cache_dir,
            dataset,
            k1=float(retriever_config.get("k1", 1.5)),
            b=float(retriever_config.get("b", 0.75)),
            epsilon=float(retriever_config.get("epsilon", 0.25)),
            analyzer=retriever_config.get("analyzer", "word-lower-v1"),
        )
        index_seconds = time.perf_counter() - index_started
        search_started = time.perf_counter()
        run, query_latencies = retriever.search_with_latencies(queries, top_k=top_k)
        search_seconds = time.perf_counter() - search_started
        return RunResult(
            run=run,
            timings={"index_seconds": index_seconds, "search_seconds": search_seconds},
            query_latencies=query_latencies,
            metadata={
                **runtime_metadata("cpu"),
                **depth_stats(run, prefix="realized_depth"),
            },
        )

    if retriever_type == "dense":
        index_started = time.perf_counter()
        retriever = DenseRetriever(
            corpus=corpus,
            cache_dir=cache_dir,
            dataset=dataset,
            model_name=retriever_config["dense_model"],
            batch_size=int(retriever_config.get("batch_size", 64)),
            device=retriever_config.get("device"),
            max_seq_length=optional_int(retriever_config.get("max_seq_length")),
            normalize_embeddings=bool(retriever_config.get("normalize_embeddings", True)),
        )
        index_seconds = time.perf_counter() - index_started
        search_started = time.perf_counter()
        run, query_latencies = retriever.search_with_latencies(
            queries,
            top_k=top_k,
            index_backend=retriever_config.get("index_backend", "numpy"),
        )
        search_seconds = time.perf_counter() - search_started
        return RunResult(
            run=run,
            timings={"index_seconds": index_seconds, "search_seconds": search_seconds},
            query_latencies=query_latencies,
            metadata={
                **runtime_metadata(retriever.device),
                "max_seq_length": retriever.max_seq_length,
                "normalize_embeddings": retriever.normalize_embeddings,
                **depth_stats(run, prefix="realized_depth"),
            },
        )

    if retriever_type == "hybrid":
        bm25_result = build_run(
            {
                "type": "bm25",
                "top_k": int(retriever_config.get("bm25_top_k", top_k)),
                "analyzer": retriever_config.get("analyzer", "word-lower-v1"),
                "k1": retriever_config.get("k1", 1.5),
                "b": retriever_config.get("b", 0.75),
                "epsilon": retriever_config.get("epsilon", 0.25),
            },
            corpus,
            queries,
            dataset,
            cache_dir,
        )
        dense_result = build_run(
            {
                "type": "dense",
                "dense_model": retriever_config["dense_model"],
                "top_k": int(retriever_config.get("dense_top_k", top_k)),
                "batch_size": retriever_config.get("batch_size", 64),
                "device": retriever_config.get("device"),
                "index_backend": retriever_config.get("index_backend", "numpy"),
                "max_seq_length": retriever_config.get("max_seq_length"),
                "normalize_embeddings": retriever_config.get("normalize_embeddings", True),
            },
            corpus,
            queries,
            dataset=dataset,
            cache_dir=cache_dir,
        )
        combine_started = time.perf_counter()
        run, combine_latencies = hybrid_rrf_with_latencies(
            bm25_runs=bm25_result.run,
            dense_runs=dense_result.run,
            query_ids=list(queries.keys()),
            top_k=top_k,
            rrf_k=int(retriever_config.get("rrf_k", 60)),
            bm25_weight=float(retriever_config.get("bm25_weight", 1.0)),
            dense_weight=float(retriever_config.get("dense_weight", 1.0)),
        )
        combine_seconds = time.perf_counter() - combine_started
        query_latencies = combine_query_latencies(
            bm25_result.query_latencies,
            dense_result.query_latencies,
            combine_latencies,
        )
        return RunResult(
            run=run,
            timings={
                "index_seconds": bm25_result.timings.get("index_seconds", 0.0)
                + dense_result.timings.get("index_seconds", 0.0),
                "search_seconds": bm25_result.timings.get("search_seconds", 0.0)
                + dense_result.timings.get("search_seconds", 0.0)
                + combine_seconds,
                "bm25_search_seconds": bm25_result.timings.get("search_seconds", 0.0),
                "dense_search_seconds": dense_result.timings.get("search_seconds", 0.0),
                "rrf_seconds": combine_seconds,
            },
            query_latencies=query_latencies,
            metadata={
                **runtime_metadata(dense_result.metadata.get("device", "")),
                **depth_stats(run, prefix="realized_depth"),
                "bm25_realized_depth_mean": depth_stats(bm25_result.run, prefix="bm25_realized_depth")[
                    "bm25_realized_depth_mean"
                ],
                "dense_realized_depth_mean": depth_stats(dense_result.run, prefix="dense_realized_depth")[
                    "dense_realized_depth_mean"
                ],
            },
        )

    if retriever_type == "rerank":
        base_config = dict(retriever_config.get("base", {"type": "bm25", "top_k": top_k}))
        base_result = build_run(base_config, corpus, queries, dataset, cache_dir)
        model_started = time.perf_counter()
        model = CrossEncoder(retriever_config["reranker_model"], device=retriever_config.get("device"))
        max_seq_length = retriever_config.get("max_seq_length")
        if max_seq_length:
            model.max_length = int(max_seq_length)
        model_load_seconds = time.perf_counter() - model_started
        rerank_started = time.perf_counter()
        run, rerank_latencies = rerank_with_latencies(
            queries=queries,
            corpus=corpus,
            base_runs=base_result.run,
            model=model,
            rerank_top_k=int(retriever_config.get("rerank_top_k", 100)),
            keep_top_k=top_k,
            batch_size=int(retriever_config.get("batch_size", 32)),
        )
        rerank_seconds = time.perf_counter() - rerank_started
        return RunResult(
            run=run,
            timings={
                "index_seconds": base_result.timings.get("index_seconds", 0.0),
                "model_load_seconds": base_result.timings.get("model_load_seconds", 0.0) + model_load_seconds,
                "search_seconds": base_result.timings.get("search_seconds", 0.0) + rerank_seconds,
                "base_search_seconds": base_result.timings.get("search_seconds", 0.0),
                "rerank_seconds": rerank_seconds,
            },
            query_latencies=combine_query_latencies(base_result.query_latencies, rerank_latencies),
            metadata={
                **base_result.metadata,
                **depth_stats(run, prefix="realized_depth"),
            },
        )

    raise ValueError(f"Unknown retriever type: {retriever_type}")


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(file)
        return json.load(file)


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    existing_rows: list[dict[str, str]] = []
    if results_path.exists():
        with results_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            old_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        fieldnames = old_fieldnames + [field for field in fieldnames if field not in old_fieldnames]

    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow(existing_row)
        writer.writerow(row)


def first_present(primary: dict[str, Any], secondary: dict[str, Any], key: str) -> Any:
    if key in primary and primary[key] not in (None, ""):
        return primary[key]
    if key in secondary and secondary[key] not in (None, ""):
        return secondary[key]
    return ""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def combine_query_latencies(*latency_lists: list[float]) -> list[float]:
    non_empty = [latencies for latencies in latency_lists if latencies]
    if not non_empty:
        return []
    expected_length = len(non_empty[0])
    if any(len(latencies) != expected_length for latencies in non_empty):
        return []
    return [sum(latencies[index] for latencies in non_empty) for index in range(expected_length)]


def depth_stats(run: dict[str, dict[str, float]], prefix: str) -> dict[str, float]:
    depths = [float(len(scores)) for scores in run.values()]
    if not depths:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_mean": sum(depths) / len(depths),
        f"{prefix}_p50": percentile(depths, 50),
        f"{prefix}_p95": percentile(depths, 95),
    }


def runtime_metadata(device: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {"device": device}
    try:
        import torch

        metadata["torch_num_threads"] = torch.get_num_threads()
        metadata["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except Exception:
        metadata["torch_num_threads"] = ""
        metadata["torch_num_interop_threads"] = ""
    return metadata


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def write_artifacts(
    artifacts_dir: Path,
    run_id: str,
    run: dict[str, dict[str, float]],
    per_query: dict[str, dict[str, float]],
    config: dict[str, Any],
    row_metadata: dict[str, Any],
) -> dict[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_path = artifacts_dir / f"{run_id}.run.jsonl"
    per_query_path = artifacts_dir / f"{run_id}.per_query.csv"
    metadata_path = artifacts_dir / f"{run_id}.metadata.json"

    with run_path.open("w", encoding="utf-8") as file:
        for query_id, scores in run.items():
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                file.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "doc_id": doc_id,
                            "rank": rank,
                            "score": score,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    metric_fields = sorted({metric for metrics in per_query.values() for metric in metrics})
    with per_query_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["query_id", *metric_fields], lineterminator="\n")
        writer.writeheader()
        for query_id in sorted(per_query):
            writer.writerow({"query_id": query_id, **per_query[query_id]})

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump({"config": config, "metadata": row_metadata}, file, indent=2, sort_keys=True)

    return {
        "run_artifact_path": str(run_path),
        "per_query_metrics_path": str(per_query_path),
        "metadata_path": str(metadata_path),
    }


def make_run_id(run_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", run_name).strip("-").lower() or "run"
    return f"{timestamp}_{slug}"


if __name__ == "__main__":
    main()
