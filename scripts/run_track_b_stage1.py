from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

from rag_bench.load import corpus_text
from rag_bench.multihop import (
    chunk_corpus,
    evaluate_evidence_per_query,
    file_sha256,
    load_multihop_dataset,
    mean_metrics,
)
from rag_bench.rerank import _build_reranked_run
from rag_bench.retrievers import BM25Retriever, DenseRetriever
from rag_bench.run import percentile, runtime_metadata, write_artifacts


BM25_CONFIG = {
    "analyzer": "word-lower-stop-porter-v1",
    "k1": 1.2,
    "b": 0.9,
    "epsilon": 0.25,
}
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
DENSE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
CUTOFFS = [1, 3, 5, 10, 20]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Track B Stage 1 evidence retrieval on MultiHop-RAG."
    )
    parser.add_argument("--stage", choices=["all", "bm25", "dense", "rerank"], default="all")
    parser.add_argument("--data-dir", default="data/multihop_rag")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--results", default="results/track_b_stage1.csv")
    parser.add_argument("--artifacts-dir", default="results/track_b_artifacts")
    parser.add_argument("--chunk-words", type=int, default=256)
    parser.add_argument("--overlap-words", type=int, default=64)
    parser.add_argument(
        "--chunking-strategy",
        choices=["fixed", "structural", "semantic"],
        default="fixed",
    )
    parser.add_argument("--semantic-break-percentile", type=float, default=15.0)
    parser.add_argument("--semantic-min-chunk-words", type=int, default=64)
    parser.add_argument("--retrieval-top-k", type=int, default=100)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--rerank-query-group-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-from-checkpoint",
        action="store_true",
        help="Rewrite a reranker row from a complete total-key checkpoint.",
    )
    args = parser.parse_args()

    positive_values = {
        "chunk_words": args.chunk_words,
        "retrieval_top_k": args.retrieval_top_k,
        "rerank_top_k": args.rerank_top_k,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "rerank_batch_size": args.rerank_batch_size,
        "rerank_query_group_size": args.rerank_query_group_size,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        parser.error(f"These arguments must be positive: {', '.join(invalid)}")
    if args.overlap_words < 0 or args.overlap_words >= args.chunk_words:
        parser.error("--overlap-words must be in [0, chunk-words).")
    if args.chunking_strategy != "fixed" and args.overlap_words:
        parser.error("Non-fixed chunking requires --overlap-words 0.")
    if args.chunking_strategy == "semantic" and not 0 <= args.semantic_break_percentile <= 100:
        parser.error("--semantic-break-percentile must be between 0 and 100.")
    if args.chunking_strategy == "semantic" and not (
        0 < args.semantic_min_chunk_words <= args.chunk_words
    ):
        parser.error("--semantic-min-chunk-words must be in [1, chunk-words].")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive.")
    if args.refresh_from_checkpoint and args.stage != "rerank":
        parser.error("--refresh-from-checkpoint requires --stage rerank.")

    if args.max_queries:
        if args.results == "results/track_b_stage1.csv":
            args.results = "results/track_b_smoke_stage1.csv"
        if args.artifacts_dir == "results/track_b_artifacts":
            args.artifacts_dir = "results/track_b_smoke_artifacts"

    dataset = load_multihop_dataset(
        args.data_dir, max_queries=args.max_queries
    )
    chunking_started = time.perf_counter()
    semantic_encoder = None
    chunking_device = ""
    if args.chunking_strategy == "semantic":
        semantic_encoder = SentenceTransformer(
            DENSE_MODEL,
            revision=DENSE_REVISION,
            device=args.device,
        )
        semantic_encoder.max_seq_length = args.max_seq_length
        chunking_device = str(semantic_encoder.device)
    chunks, chunk_to_doc = chunk_corpus(
        dataset.corpus,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        strategy=args.chunking_strategy,
        semantic_encoder=semantic_encoder,
        semantic_break_percentile=args.semantic_break_percentile,
        semantic_min_chunk_words=args.semantic_min_chunk_words,
    )
    chunking_seconds = time.perf_counter() - chunking_started
    chunk_artifact_path = persist_chunk_artifact(
        Path(args.cache_dir),
        chunks,
        {
            "corpus_sha256": dataset.source_hashes["corpus_sha256"],
            "chunk_words": args.chunk_words,
            "overlap_words": args.overlap_words,
            "chunking_strategy": args.chunking_strategy,
            "semantic_break_percentile": (
                args.semantic_break_percentile
                if args.chunking_strategy == "semantic"
                else ""
            ),
            "semantic_min_chunk_words": (
                args.semantic_min_chunk_words
                if args.chunking_strategy == "semantic"
                else ""
            ),
            "chunking_model": (
                DENSE_MODEL if args.chunking_strategy == "semantic" else ""
            ),
            "chunking_model_revision": (
                DENSE_REVISION if args.chunking_strategy == "semantic" else ""
            ),
        },
    )
    chunk_artifact_sha256 = file_sha256(chunk_artifact_path)
    output_path = Path(args.results)
    artifacts_dir = Path(args.artifacts_dir)
    rows = latest_rows(output_path)
    runs: dict[str, dict[str, dict[str, float]]] = {}

    stages = ["bm25", "dense", "rerank"] if args.stage == "all" else [args.stage]
    for stage in stages:
        run_name = make_run_name(stage, args)
        prior = rows.get(run_name)
        if (
            prior
            and not args.force
            and not args.refresh_from_checkpoint
            and Path(prior["run_artifact_path"]).exists()
        ):
            print(f"SKIP {run_name}")
            runs[stage] = read_run(Path(prior["run_artifact_path"]))
            continue

        print(f"RUN {run_name}")
        config = stage_config(stage, args)
        config.update(dataset.source_hashes)
        checkpoint_path: Path | None = None
        base_row: dict[str, str] | None = None
        started = time.perf_counter()
        if stage == "bm25":
            model_started = time.perf_counter()
            retriever = BM25Retriever(
                corpus=chunks,
                cache_dir=args.cache_dir,
                dataset=chunk_dataset_name(args),
                **BM25_CONFIG,
            )
            model_load_seconds = time.perf_counter() - model_started
            search_started = time.perf_counter()
            run, latencies = retriever.search_with_latencies(
                dataset.queries, args.retrieval_top_k
            )
            search_seconds = time.perf_counter() - search_started
            device = "cpu"
        elif stage == "dense":
            model_started = time.perf_counter()
            retriever = DenseRetriever(
                corpus=chunks,
                cache_dir=args.cache_dir,
                dataset=chunk_dataset_name(args),
                model_name=DENSE_MODEL,
                revision=DENSE_REVISION,
                batch_size=args.batch_size,
                device=args.device,
                max_seq_length=args.max_seq_length,
                normalize_embeddings=True,
            )
            model_load_seconds = time.perf_counter() - model_started
            search_started = time.perf_counter()
            run, latencies = retriever.search_with_latencies(
                dataset.queries,
                args.retrieval_top_k,
                index_backend="numpy",
            )
            search_seconds = time.perf_counter() - search_started
            device = retriever.device
        else:
            dense_run_name = make_run_name("dense", args)
            base_row = rows.get(dense_run_name)
            if base_row is None:
                raise ValueError(
                    f"Required base run {dense_run_name!r} is missing. Run --stage dense first."
                )
            base_artifact_path = Path(base_row["run_artifact_path"])
            base_artifact_sha256 = file_sha256(base_artifact_path)
            dense_run = runs.get("dense") or read_run(base_artifact_path)
            config.update(
                {
                    "base_run_id": base_row["run_id"],
                    "base_config_sha256": base_row["config_sha256"],
                    "base_run_artifact_sha256": base_artifact_sha256,
                }
            )
            model_started = time.perf_counter()
            model = CrossEncoder(
                RERANKER_MODEL,
                revision=RERANKER_REVISION,
                max_length=args.max_seq_length,
                device=args.device,
                trust_remote_code=True,
            )
            model_load_seconds = time.perf_counter() - model_started
            search_started = time.perf_counter()
            checkpoint_path = rerank_checkpoint_path(
                Path(args.cache_dir), run_name, config
            )
            if args.refresh_from_checkpoint and not checkpoint_path.exists():
                raise ValueError(
                    f"No complete checkpoint exists for refresh: {checkpoint_path}"
                )
            if args.force and not args.refresh_from_checkpoint and checkpoint_path.exists():
                checkpoint_path.unlink()
            run, latencies = rerank_with_checkpoint(
                queries=dataset.queries,
                corpus=chunks,
                base_runs=dense_run,
                model=model,
                rerank_top_k=args.rerank_top_k,
                keep_top_k=args.retrieval_top_k,
                batch_size=args.rerank_batch_size,
                checkpoint_path=checkpoint_path,
                query_group_size=args.rerank_query_group_size,
            )
            # Checkpoint resumes span invocations, so the complete amortized latency
            # ledger is a better denominator than this invocation's wall clock.
            search_seconds = sum(latencies)
            device = str(model.device)

        per_query = evaluate_evidence_per_query(
            evidence=dataset.evidence,
            runs=run,
            chunk_corpus=chunks,
            chunk_to_doc=chunk_to_doc,
            cutoffs=CUTOFFS,
        )
        metrics = mean_metrics(per_query)
        realized_depths = [len(ranked) for ranked in run.values()]
        run_id = make_run_id(run_name, config)
        row: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": run_id,
            "run_name": run_name,
            "dataset": "MultiHop-RAG",
            "retriever_type": stage,
            "num_queries_retrieved": len(dataset.queries),
            "num_answerable_queries_scored": len(per_query),
            "num_null_queries": len(dataset.null_query_ids),
            "corpus_documents": len(dataset.corpus),
            "corpus_chunks": len(chunks),
            "chunk_words": args.chunk_words,
            "overlap_words": args.overlap_words,
            "chunk_unit": "whitespace_word",
            "chunking_strategy": args.chunking_strategy,
            "chunking_seconds": round(chunking_seconds, 3),
            "chunking_model": DENSE_MODEL if semantic_encoder is not None else "",
            "chunking_model_revision": (
                DENSE_REVISION if semantic_encoder is not None else ""
            ),
            "chunking_device": chunking_device,
            "chunk_artifact_path": str(chunk_artifact_path),
            "chunk_artifact_sha256": chunk_artifact_sha256,
            "semantic_break_percentile": (
                args.semantic_break_percentile if semantic_encoder is not None else ""
            ),
            "semantic_min_chunk_words": (
                args.semantic_min_chunk_words if semantic_encoder is not None else ""
            ),
            "corpus_text": "title+source+published_at+body",
            "chunks_per_document": round(len(chunks) / len(dataset.corpus), 6),
            "retrieval_top_k": args.retrieval_top_k,
            "realized_depth_mean": round(
                sum(realized_depths) / len(realized_depths), 6
            ),
            "realized_depth_min": min(realized_depths),
            "realized_depth_p50": round(percentile(realized_depths, 50), 3),
            "realized_depth_p95": round(percentile(realized_depths, 95), 3),
            "realized_depth_max": max(realized_depths),
            "rerank_top_k": args.rerank_top_k if stage == "rerank" else "",
            "dense_batch_size": args.batch_size if stage in {"dense", "rerank"} else "",
            "rerank_batch_size": args.rerank_batch_size if stage == "rerank" else "",
            "rerank_query_group_size": (
                args.rerank_query_group_size if stage == "rerank" else ""
            ),
            "bm25_analyzer": BM25_CONFIG["analyzer"] if stage == "bm25" else "",
            "bm25_k1": BM25_CONFIG["k1"] if stage == "bm25" else "",
            "bm25_b": BM25_CONFIG["b"] if stage == "bm25" else "",
            "dense_model": DENSE_MODEL if stage in {"dense", "rerank"} else "",
            "dense_revision": DENSE_REVISION if stage in {"dense", "rerank"} else "",
            "reranker_model": RERANKER_MODEL if stage == "rerank" else "",
            "reranker_revision": RERANKER_REVISION if stage == "rerank" else "",
            "trust_remote_code": stage == "rerank",
            "base_run_id": base_row["run_id"] if base_row else "",
            "base_run_name": base_row["run_name"] if base_row else "",
            "base_config_sha256": base_row["config_sha256"] if base_row else "",
            "base_run_artifact_sha256": (
                config["base_run_artifact_sha256"] if base_row else ""
            ),
            "max_seq_length": args.max_seq_length if stage in {"dense", "rerank"} else "",
            "normalize_embeddings": stage in {"dense", "rerank"},
            "index_backend": "numpy_exact" if stage in {"dense", "rerank"} else "bm25_exact",
            "device": device,
            "model_load_or_index_seconds": round(model_load_seconds, 3),
            "search_seconds": round(search_seconds, 3),
            "wall_seconds": round(time.perf_counter() - started, 3),
            "throughput_ms_per_query": round(search_seconds / len(dataset.queries) * 1000, 3),
            "query_latency_p50_ms": round(percentile(latencies, 50) * 1000, 3),
            "query_latency_p95_ms": round(percentile(latencies, 95) * 1000, 3),
            "latency_method": (
                "checkpointed_cross_query_batch_amortized"
                if stage == "rerank"
                else "single_query_sequential_warm_model"
            ),
            "queries_sha256": dataset.source_hashes["queries_sha256"],
            "corpus_sha256": dataset.source_hashes["corpus_sha256"],
            "config_sha256": config_hash(config),
            "python_version": platform.python_version(),
            "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
            **runtime_metadata(device),
            **{name: round(value, 6) for name, value in metrics.items()},
        }
        artifact_paths = write_artifacts(
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            run=run,
            per_query=per_query,
            config=config,
            row_metadata=row,
        )
        row.update(artifact_paths)
        upsert_result(output_path, row)
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()
        rows[run_name] = {key: str(value) for key, value in row.items()}
        runs[stage] = run
        print(json.dumps(row, indent=2, sort_keys=True))


def stage_config(stage: str, args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "stage": stage,
        "data_dir": args.data_dir,
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "chunk_unit": "whitespace_word",
        "corpus_text": "title+source+published_at+body",
        "retrieval_top_k": args.retrieval_top_k,
        "cutoffs": CUTOFFS,
    }
    if args.chunking_strategy != "fixed":
        config["chunking_strategy"] = args.chunking_strategy
    if args.chunking_strategy == "semantic":
        config.update(
            {
                "chunking_model": DENSE_MODEL,
                "chunking_model_revision": DENSE_REVISION,
                "semantic_break_percentile": args.semantic_break_percentile,
                "semantic_min_chunk_words": args.semantic_min_chunk_words,
            }
        )
    if stage == "bm25":
        config.update(BM25_CONFIG)
    else:
        config.update(
            {
                "dense_model": DENSE_MODEL,
                "dense_revision": DENSE_REVISION,
                "dense_batch_size": args.batch_size,
                "max_seq_length": args.max_seq_length,
                "normalize_embeddings": True,
                "index_backend": "numpy_exact",
            }
        )
    if stage == "rerank":
        config.update(
            {
                "reranker_model": RERANKER_MODEL,
                "reranker_revision": RERANKER_REVISION,
                "rerank_top_k": args.rerank_top_k,
                "rerank_batch_size": args.rerank_batch_size,
                "rerank_query_group_size": args.rerank_query_group_size,
                "trust_remote_code": True,
            }
        )
    return config


def make_run_name(stage: str, args: argparse.Namespace) -> str:
    prefix = f"smoke{args.max_queries}_" if args.max_queries else ""
    strategy = "" if args.chunking_strategy == "fixed" else f"{args.chunking_strategy}_"
    suffix = f"{strategy}w{args.chunk_words}_o{args.overlap_words}"
    names = {
        "bm25": f"{prefix}multihop_bm25_stem_{suffix}",
        "dense": f"{prefix}multihop_dense_bge_{suffix}",
        "rerank": f"{prefix}multihop_dense_bge_rerank{args.rerank_top_k}_{suffix}",
    }
    return names[stage]


def chunk_dataset_name(args: argparse.Namespace) -> str:
    if args.chunking_strategy == "fixed":
        return f"multihop_rag_w{args.chunk_words}_o{args.overlap_words}"
    return (
        f"multihop_rag_{args.chunking_strategy}_"
        f"w{args.chunk_words}_o{args.overlap_words}"
    )


def make_run_id(run_name: str, config: dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{run_name}_{config_hash(config)[:8]}"


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rerank_checkpoint_path(
    cache_dir: Path, run_name: str, config: dict[str, Any]
) -> Path:
    return (
        cache_dir
        / "track_b_rerank"
        / f"{run_name}_{config_hash(config)[:12]}.jsonl"
    )


def persist_chunk_artifact(
    cache_dir: Path,
    chunks: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    artifact_dir = cache_dir / "track_b_chunks"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"chunks_{config_hash(config)[:16]}.jsonl"
    if path.exists():
        return path
    temporary_path = path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for chunk_id, chunk in chunks.items():
            file.write(
                json.dumps(
                    {"chunk_id": chunk_id, "chunk": chunk},
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
    temporary_path.replace(path)
    return path


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            run.setdefault(item["query_id"], {})[item["doc_id"]] = float(item["score"])
    return run


def rerank_with_checkpoint(
    queries: dict[str, str],
    corpus: dict[str, dict[str, Any]],
    base_runs: dict[str, dict[str, float]],
    model: CrossEncoder,
    rerank_top_k: int,
    keep_top_k: int,
    batch_size: int,
    checkpoint_path: Path,
    query_group_size: int = 16,
) -> tuple[dict[str, dict[str, float]], list[float]]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, float]] = {}
    latency_by_query: dict[str, float] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                runs[item["query_id"]] = {
                    doc_id: float(score) for doc_id, score in item["scores"].items()
                }
                latency_by_query[item["query_id"]] = float(item["latency_seconds"])

    pending = [query_id for query_id in queries if query_id not in runs]
    for group_start in tqdm(
        range(0, len(pending), query_group_size), desc="Checkpointed reranking"
    ):
        group = pending[group_start : group_start + query_group_size]
        candidates_by_query: dict[str, list[tuple[str, float]]] = {}
        pairs: list[tuple[str, str]] = []
        pair_counts: list[int] = []
        for query_id in group:
            candidates = sorted(
                base_runs[query_id].items(), key=lambda item: (-item[1], item[0])
            )
            candidates_by_query[query_id] = candidates
            head = candidates[: min(rerank_top_k, len(candidates), keep_top_k)]
            pair_counts.append(len(head))
            pairs.extend(
                (queries[query_id], corpus_text(corpus[doc_id])) for doc_id, _ in head
            )

        group_started = time.perf_counter()
        predicted = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        group_seconds = time.perf_counter() - group_started
        amortized_latency = group_seconds / len(group)
        offset = 0
        checkpoint_rows: list[dict[str, Any]] = []
        for query_id, pair_count in zip(group, pair_counts, strict=True):
            candidates = candidates_by_query[query_id]
            head = candidates[:pair_count]
            scores = {
                doc_id: float(score)
                for (doc_id, _), score in zip(
                    head, predicted[offset : offset + pair_count], strict=True
                )
            }
            offset += pair_count
            run = _build_reranked_run(
                candidates=candidates,
                rerank_scores=scores,
                rerank_top_k=pair_count,
                keep_top_k=keep_top_k,
            )
            runs[query_id] = run
            latency_by_query[query_id] = amortized_latency
            checkpoint_rows.append(
                {
                    "query_id": query_id,
                    "latency_seconds": amortized_latency,
                    "scores": run,
                }
            )
        with checkpoint_path.open("a", encoding="utf-8") as file:
            for item in checkpoint_rows:
                file.write(json.dumps(item, sort_keys=True) + "\n")

    return runs, [latency_by_query[query_id] for query_id in queries]


def latest_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        return {row["run_name"]: row for row in csv.DictReader(file)}


def upsert_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = latest_rows(path)
    rows[str(row["run_name"])] = {key: value for key, value in row.items()}
    fieldnames: list[str] = []
    for existing in rows.values():
        fieldnames.extend(key for key in existing if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows.values())


if __name__ == "__main__":
    main()
