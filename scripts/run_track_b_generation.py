from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from rag_bench.generation import (
    ANSWER_NORMALIZER_VERSION,
    ABSTENTION_INSTRUCTION_ON,
    DEFAULT_ABSTENTION_PATTERNS,
    EVALUATION_PARTITION_VERSION,
    EVALUATION_SAMPLER_VERSION,
    OllamaConfig,
    PROMPT_DEVELOPMENT_QUERY_COUNT,
    RESPONSE_PARSER_VERSION,
    abstention_metrics,
    exact_match,
    is_abstention,
    ollama_generate,
    ollama_model_metadata,
    parse_answer_and_claim,
    prompt_hash,
    token_f1,
    stable_query_sample,
)
from rag_bench.multihop import (
    Evidence,
    chunk_corpus,
    file_sha256,
    load_multihop_dataset,
    normalize_space,
)
from rag_bench.run import percentile


PROMPT_TEMPLATE = """Answer this multi-hop question using the question and evidence below.
Combine information across all evidence blocks. Titles, sources, and dates are part of the
evidence. A block may paraphrase a premise from the question instead of repeating every
detail, so identify the entity or answer supported by the blocks together. Do not use
outside knowledge.
For yes/no, comparison, and consistency questions, make the requested comparison and
answer Yes or No. Do not abstain merely because the relationship requires combining
multiple blocks.
{abstention_instruction}
Return exactly two lines:
Answer: <the concise answer>
Claim: <one complete sentence of at most 25 words that explicitly states and supports that answer>

Evidence:
{context}

Question: {question}"""
CLOSED_BOOK_PROMPT_TEMPLATE = """Answer this multi-hop question from your own knowledge.
For yes/no, comparison, and consistency questions, make the requested comparison and
answer Yes or No. If you do not know, answer "Insufficient information."
Return exactly two lines:
Answer: <the concise answer>
Claim: <one complete sentence of at most 25 words that explicitly states and supports that answer>

Question: {question}"""
CONTEXT_BUILDER_VERSION = "oracle-facts-metadata-retrieved-ranked-chunks-v1"
RETRIEVED_CONTEXT_BUILDER_VERSION = (
    "retrieved-ranked-chunks-balanced-total-word-budget-v3"
)
GREEDY_CONTEXT_BUILDER_VERSION = "retrieved-ranked-chunks-greedy-total-word-budget-v1"
CLOSED_BOOK_CONTEXT_BUILDER_VERSION = "closed-book-no-context-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pinned local Track B generation pass.")
    parser.add_argument(
        "--mode", choices=["oracle", "retrieved", "closed_book"], required=True
    )
    parser.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--retriever-run-name")
    parser.add_argument("--retrieval-runs", default="results/track_b_stage1.csv")
    parser.add_argument("--retrieved-top-k", type=int, default=5)
    parser.add_argument("--max-context-words", type=int, default=1280)
    parser.add_argument(
        "--context-packing",
        choices=["balanced", "greedy"],
        default="balanced",
        help="Use greedy packing for matched-budget chunking comparisons.",
    )
    parser.add_argument(
        "--evidence-position",
        choices=["natural", "first", "middle", "last"],
        default="natural",
    )
    parser.add_argument(
        "--abstention-instruction", choices=["on", "off"], default="on"
    )
    parser.add_argument("--data-dir", default="data/multihop_rag")
    parser.add_argument("--artifacts-dir", default="results/track_b_generation_artifacts")
    parser.add_argument("--results", default="results/track_b_generation_runs.csv")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--evaluation-sample-size", type=int)
    parser.add_argument("--evaluation-sample-seed", type=int, default=13)
    args = parser.parse_args()

    positive_values = {
        "retrieved_top_k": args.retrieved_top_k,
        "max_context_words": args.max_context_words,
        "context_window": args.context_window,
        "max_new_tokens": args.max_new_tokens,
        "repeat": args.repeat,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        parser.error(f"These arguments must be positive: {', '.join(invalid)}")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive.")
    if args.evaluation_sample_size is not None and args.evaluation_sample_size <= 0:
        parser.error("--evaluation-sample-size must be positive.")
    if args.max_queries is not None and args.evaluation_sample_size is not None:
        parser.error("--max-queries and --evaluation-sample-size are mutually exclusive.")

    if args.max_queries is not None:
        if args.results == "results/track_b_generation_runs.csv":
            args.results = "results/track_b_smoke_generation_runs.csv"
        if args.artifacts_dir == "results/track_b_generation_artifacts":
            args.artifacts_dir = "results/track_b_smoke_generation_artifacts"

    if args.mode == "retrieved" and not args.retriever_run_name:
        parser.error("--retriever-run-name is required in retrieved mode.")

    dataset = load_multihop_dataset(args.data_dir)
    chunks: dict[str, dict[str, Any]] = {}
    ranked_runs: dict[str, dict[str, float]] = {}
    retriever_row: dict[str, str] | None = None
    if args.mode == "retrieved":
        retriever_row = find_run(Path(args.retrieval_runs), args.retriever_run_name)
        ranked_runs = read_run(Path(retriever_row["run_artifact_path"]))
        chunking_strategy = retriever_row.get("chunking_strategy") or "fixed"
        chunk_artifact_path = Path(retriever_row.get("chunk_artifact_path") or "")
        if chunk_artifact_path.is_file():
            expected_sha256 = retriever_row.get("chunk_artifact_sha256") or ""
            actual_sha256 = file_sha256(chunk_artifact_path)
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise ValueError(
                    "Chunk artifact hash does not match the retrieval row: "
                    f"{chunk_artifact_path}"
                )
            chunks = read_chunks(chunk_artifact_path)
        else:
            if chunking_strategy == "semantic":
                raise ValueError(
                    "Semantic generation requires its persisted chunk artifact. "
                    "Rerun the corresponding Stage 1 retrieval first."
                )
            chunks, _ = chunk_corpus(
                dataset.corpus,
                chunk_words=int(retriever_row["chunk_words"]),
                overlap_words=int(retriever_row["overlap_words"]),
                strategy=chunking_strategy,
            )

    candidate_query_ids = (
        dataset.answerable_query_ids if args.mode == "oracle" else list(dataset.queries)
    )
    prompt_development_ids = dataset.answerable_query_ids[
        :PROMPT_DEVELOPMENT_QUERY_COUNT
    ]
    if args.max_queries is not None:
        query_ids = candidate_query_ids[: args.max_queries]
        evaluation_partition_version = "development-smoke-included-v1"
        prompt_development_excluded = 0
    else:
        prompt_development_set = set(prompt_development_ids)
        query_ids = [
            query_id
            for query_id in candidate_query_ids
            if query_id not in prompt_development_set
        ]
        evaluation_partition_version = EVALUATION_PARTITION_VERSION
        prompt_development_excluded = len(prompt_development_ids)
        if args.evaluation_sample_size is not None:
            query_ids = stable_query_sample(
                query_ids,
                size=args.evaluation_sample_size,
                seed=args.evaluation_sample_seed,
            )

    if args.max_queries is not None:
        query_scope = f"first_{args.max_queries}"
    elif args.evaluation_sample_size is not None:
        query_scope = (
            f"evaluation_hash_sample_{args.evaluation_sample_size}_"
            f"seed_{args.evaluation_sample_seed}"
        )
    else:
        query_scope = (
            "evaluation_answerable_excluding_prompt_development"
            if args.mode == "oracle"
            else "evaluation_all_excluding_prompt_development"
        )

    ollama_config = OllamaConfig(
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        context_window=args.context_window,
        max_new_tokens=args.max_new_tokens,
    )
    model_metadata = ollama_model_metadata(ollama_config)
    model_details = model_metadata.get("details", {})
    runtime_platform = platform.platform()
    runtime_python = platform.python_version()
    if args.mode == "closed_book":
        template = CLOSED_BOOK_PROMPT_TEMPLATE
    else:
        template = PROMPT_TEMPLATE.replace(
            "{abstention_instruction}",
            ABSTENTION_INSTRUCTION_ON if args.abstention_instruction == "on" else "",
        )
    experiment = {
        "mode": args.mode,
        "retriever_run_id": retriever_row["run_id"] if retriever_row else args.mode,
        "retriever_run_name": args.retriever_run_name or args.mode,
        "retriever_config_sha256": (
            retriever_row["config_sha256"] if retriever_row else args.mode
        ),
        "retriever_artifact_sha256": (
            file_sha256(Path(retriever_row["run_artifact_path"]))
            if retriever_row
            else args.mode
        ),
        "retrieved_top_k": args.retrieved_top_k if args.mode == "retrieved" else "",
        "max_context_words": args.max_context_words if args.mode == "retrieved" else "",
        "evidence_position": args.evidence_position if args.mode == "retrieved" else "natural",
        "model": args.model,
        "model_digest": model_metadata.get("manifest_digest", ""),
        "model_family": model_details.get("family", ""),
        "model_parameter_size": model_details.get("parameter_size", ""),
        "quantization_level": model_details.get("quantization_level", ""),
        "ollama_version": model_metadata.get("ollama_version", ""),
        "temperature": args.temperature,
        "seed": args.seed,
        "context_window": args.context_window,
        "max_new_tokens": args.max_new_tokens,
        "abstention_instruction": (
            args.abstention_instruction if args.mode != "closed_book" else "closed_book"
        ),
        "prompt_sha256": prompt_hash(template),
        "answer_normalizer_version": ANSWER_NORMALIZER_VERSION,
        "response_parser_version": RESPONSE_PARSER_VERSION,
        "context_builder_version": (
            CLOSED_BOOK_CONTEXT_BUILDER_VERSION
            if args.mode == "closed_book"
            else (
                GREEDY_CONTEXT_BUILDER_VERSION
                if args.context_packing == "greedy"
                else RETRIEVED_CONTEXT_BUILDER_VERSION
            )
            if args.mode == "retrieved"
            else CONTEXT_BUILDER_VERSION
        ),
        "evaluation_partition_version": evaluation_partition_version,
        "prompt_development_query_count": PROMPT_DEVELOPMENT_QUERY_COUNT,
        "prompt_development_query_ids_sha256": hashlib.sha256(
            "\n".join(prompt_development_ids).encode("utf-8")
        ).hexdigest(),
        "abstention_patterns_sha256": hashlib.sha256(
            json.dumps(DEFAULT_ABSTENTION_PATTERNS).encode("utf-8")
        ).hexdigest(),
        "repeat": args.repeat,
        "query_scope": query_scope,
        "selected_query_ids_sha256": hashlib.sha256(
            "\n".join(query_ids).encode("utf-8")
        ).hexdigest(),
        "queries_sha256": dataset.source_hashes["queries_sha256"],
        "corpus_sha256": dataset.source_hashes["corpus_sha256"],
        "platform": runtime_platform,
        "python_version": runtime_python,
    }
    if args.evaluation_sample_size is not None:
        experiment.update(
            {
                "evaluation_sample_size": args.evaluation_sample_size,
                "evaluation_sample_seed": args.evaluation_sample_seed,
                "evaluation_sampler_version": EVALUATION_SAMPLER_VERSION,
            }
        )
    if args.mode == "retrieved":
        experiment["context_packing"] = args.context_packing
    run_key = hash_payload(experiment)[:16]
    run_name = f"multihop_{args.mode}_{experiment['retriever_run_name']}_{run_key}_r{args.repeat}"
    output_dir = Path(args.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{run_name}.jsonl"
    per_query_path = output_dir / f"{run_name}.per_query.csv"
    metadata_path = output_dir / f"{run_name}.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "run_key": run_key,
                "experiment": experiment,
                "prompt_template": template,
                "abstention_patterns": DEFAULT_ABSTENTION_PATTERNS,
                "model_details": model_details,
                "runtime": {
                    "platform": runtime_platform,
                    "python_version": runtime_python,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = read_generation_rows(jsonl_path)

    started = time.perf_counter()
    with jsonl_path.open("a", encoding="utf-8") as file:
        for index, query_id in enumerate(query_ids, start=1):
            if query_id in completed:
                continue
            context, context_ids = build_context(
                mode=args.mode,
                query_id=query_id,
                dataset=dataset,
                chunks=chunks,
                ranked_runs=ranked_runs,
                top_k=args.retrieved_top_k,
                max_context_words=args.max_context_words,
                evidence_position=args.evidence_position,
                context_packing=args.context_packing,
            )
            prompt = template.format(
                context=context, question=dataset.queries[query_id]
            )
            generated = ollama_generate(prompt, ollama_config)
            raw_response = str(generated["response"]).strip()
            answer, faithfulness_text, response_format_valid = parse_answer_and_claim(
                raw_response
            )
            row = {
                "query_id": query_id,
                "question": dataset.queries[query_id],
                "question_type": dataset.question_types[query_id],
                "gold_answer": dataset.answers[query_id],
                "raw_response": raw_response,
                "prediction": answer,
                "faithfulness_text": faithfulness_text,
                "response_format_valid": response_format_valid,
                "exact_match": exact_match(answer, dataset.answers[query_id]),
                "token_f1": token_f1(answer, dataset.answers[query_id]),
                "abstained": is_abstention(answer),
                "context_ids": context_ids,
                "context": context,
                "context_word_count": len(context.split()),
                "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                "total_duration_ns": generated.get("total_duration"),
                "load_duration_ns": generated.get("load_duration"),
                "prompt_eval_duration_ns": generated.get("prompt_eval_duration"),
                "eval_duration_ns": generated.get("eval_duration"),
                "prompt_eval_count": generated.get("prompt_eval_count"),
                "eval_count": generated.get("eval_count"),
            }
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            file.flush()
            completed[query_id] = row
            print(
                f"[{index}/{len(query_ids)}] {query_id} "
                f"EM={row['exact_match']:.0f} F1={row['token_f1']:.3f}"
            )

    ordered_rows = [completed[query_id] for query_id in query_ids]
    write_per_query(per_query_path, ordered_rows)
    answerable_rows = [
        row for row in ordered_rows if row["question_type"] != "null_query"
    ]
    answerable_types = sorted({row["question_type"] for row in answerable_rows})
    metrics_by_type: dict[str, float] = {}
    for question_type in answerable_types:
        type_rows = [
            row for row in answerable_rows if row["question_type"] == question_type
        ]
        metrics_by_type[f"exact_match_{question_type}"] = mean(
            float(row["exact_match"]) for row in type_rows
        )
        metrics_by_type[f"token_f1_{question_type}"] = mean(
            float(row["token_f1"]) for row in type_rows
        )
    predictions = {row["query_id"]: row["prediction"] for row in ordered_rows}
    total_durations = [
        float(row["total_duration_ns"]) / 1_000_000_000
        for row in ordered_rows
        if row.get("total_duration_ns") is not None
    ]
    valid_component_rows = [
        row for row in ordered_rows if duration_components_are_valid(row)
    ]
    load_durations = [
        float(row["load_duration_ns"]) / 1_000_000_000
        for row in valid_component_rows
    ]
    prompt_eval_durations = [
        float(row["prompt_eval_duration_ns"]) / 1_000_000_000
        for row in valid_component_rows
    ]
    eval_durations = [
        float(row["eval_duration_ns"]) / 1_000_000_000
        for row in valid_component_rows
    ]
    prompt_eval_counts = [
        int(row["prompt_eval_count"])
        for row in ordered_rows
        if row.get("prompt_eval_count") is not None
    ]
    eval_counts = [
        int(row["eval_count"])
        for row in ordered_rows
        if row.get("eval_count") is not None
    ]
    claim_word_counts = [
        len(str(row["faithfulness_text"]).split()) for row in ordered_rows
    ]
    context_word_counts = [
        int(row.get("context_word_count", len(str(row["context"]).split())))
        for row in ordered_rows
    ]
    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_key": run_key,
        **experiment,
        "model_family": model_details.get("family", ""),
        "model_parameter_size": model_details.get("parameter_size", ""),
        "platform": runtime_platform,
        "python_version": runtime_python,
        "num_queries": len(ordered_rows),
        "num_answerable_queries": len(answerable_rows),
        "num_prompt_development_queries_excluded": prompt_development_excluded,
        "exact_match": mean(float(row["exact_match"]) for row in answerable_rows),
        "token_f1": mean(float(row["token_f1"]) for row in answerable_rows),
        "response_format_valid_rate": mean(
            float(row["response_format_valid"]) for row in ordered_rows
        ),
        "context_words_mean": mean(context_word_counts),
        "context_words_p50": percentile(context_word_counts, 50),
        "context_words_p95": percentile(context_word_counts, 95),
        "context_words_max": max(context_word_counts),
        **metrics_by_type,
        **abstention_metrics(predictions, dataset.question_types),
        "ollama_total_duration_seconds": sum(total_durations),
        "ollama_load_duration_seconds": sum(load_durations),
        "ollama_prompt_eval_duration_seconds": sum(prompt_eval_durations),
        "ollama_eval_duration_seconds": sum(eval_durations),
        "ollama_duration_component_valid_rate": (
            len(valid_component_rows) / len(ordered_rows) if ordered_rows else None
        ),
        "ollama_duration_component_invalid_count": (
            len(ordered_rows) - len(valid_component_rows)
        ),
        "prompt_eval_tokens_mean": mean(prompt_eval_counts) if prompt_eval_counts else None,
        "generated_tokens_mean": mean(eval_counts) if eval_counts else None,
        "output_cap_hit_rate": (
            mean(count >= args.max_new_tokens for count in eval_counts)
            if eval_counts
            else None
        ),
        "claim_words_p95": percentile(claim_word_counts, 95),
        "claim_words_max": max(claim_word_counts),
        "query_latency_p50_ms": percentile(total_durations, 50) * 1000,
        "query_latency_p95_ms": percentile(total_durations, 95) * 1000,
        "latency_method": "ollama_sequential_request_total_duration",
        "wall_seconds_this_invocation": round(time.perf_counter() - started, 3),
        "artifact_path": str(jsonl_path),
        "per_query_metrics_path": str(per_query_path),
        "metadata_path": str(metadata_path),
    }
    upsert_summary(Path(args.results), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_context(
    mode: str,
    query_id: str,
    dataset,
    chunks: dict[str, dict[str, Any]],
    ranked_runs: dict[str, dict[str, float]],
    top_k: int,
    max_context_words: int,
    evidence_position: str,
    context_packing: str = "balanced",
) -> tuple[str, list[str]]:
    if mode == "closed_book":
        return "", []
    if mode == "oracle":
        evidence = dataset.evidence[query_id]
        blocks = [oracle_block(dataset.corpus[item.doc_id], item) for item in evidence]
        return "\n\n".join(blocks), [item.doc_id for item in evidence]

    ranked_ids = [
        chunk_id
        for chunk_id, _ in sorted(
            ranked_runs[query_id].items(), key=lambda item: (-item[1], item[0])
        )[:top_k]
    ]
    ranked_ids = position_evidence_chunks(
        ranked_ids,
        dataset.evidence[query_id],
        chunks,
        evidence_position,
    )
    if context_packing == "greedy":
        return greedy_context(chunks, ranked_ids, max_context_words)
    if context_packing != "balanced":
        raise ValueError(f"Unknown context packing strategy: {context_packing}")
    selected: list[tuple[str, str, list[str]]] = []
    prefix_words_total = 0
    for chunk_id in ranked_ids:
        prefix = f"[{len(selected) + 1}] {chunks[chunk_id]['title']}"
        body_words = str(chunks[chunk_id]["text"]).split()
        minimum_words = len(prefix.split()) + int(bool(body_words))
        if prefix_words_total + len(selected) + minimum_words > max_context_words:
            break
        selected.append((chunk_id, prefix, body_words))
        prefix_words_total += len(prefix.split())

    allocations = balanced_word_allocations(
        [(chunk_id, len(body_words)) for chunk_id, _, body_words in selected],
        max_context_words - prefix_words_total,
    )
    blocks = [
        f"{prefix}\n{' '.join(body_words[:allocations[chunk_id]])}"
        for chunk_id, prefix, body_words in selected
    ]
    selected_ids = [chunk_id for chunk_id, _, _ in selected]
    return "\n\n".join(blocks), selected_ids


def greedy_context(
    chunks: dict[str, dict[str, Any]],
    ranked_ids: list[str],
    max_context_words: int,
) -> tuple[str, list[str]]:
    blocks: list[str] = []
    selected_ids: list[str] = []
    remaining = max_context_words
    for chunk_id in ranked_ids:
        prefix = f"[{len(blocks) + 1}] {chunks[chunk_id]['title']}"
        prefix_words = prefix.split()
        body_words = str(chunks[chunk_id]["text"]).split()
        body_budget = remaining - len(prefix_words)
        if body_budget <= 0 or not body_words:
            break
        selected_body = body_words[:body_budget]
        blocks.append(f"{prefix}\n{' '.join(selected_body)}")
        selected_ids.append(chunk_id)
        remaining -= len(prefix_words) + len(selected_body)
        if len(selected_body) < len(body_words):
            break
    return "\n\n".join(blocks), selected_ids


def balanced_word_allocations(
    lengths: list[tuple[str, int]], budget: int
) -> dict[str, int]:
    allocations = {chunk_id: 0 for chunk_id, _ in lengths}
    capacities = dict(lengths)
    stable_ids = sorted(allocations)
    while budget > 0:
        progressed = False
        for chunk_id in stable_ids:
            if allocations[chunk_id] >= capacities[chunk_id]:
                continue
            allocations[chunk_id] += 1
            budget -= 1
            progressed = True
            if budget == 0:
                break
        if not progressed:
            break
    return allocations


def duration_components_are_valid(row: dict[str, Any]) -> bool:
    fields = (
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_duration_ns",
        "eval_duration_ns",
    )
    if any(row.get(field) is None for field in fields):
        return False
    component_total = sum(float(row[field]) for field in fields[1:])
    return component_total <= float(row["total_duration_ns"])


def position_evidence_chunks(
    chunk_ids: list[str],
    evidence: tuple[Evidence, ...],
    chunks: dict[str, dict[str, Any]],
    position: str,
) -> list[str]:
    if position == "natural":
        return chunk_ids
    gold: list[str] = []
    other: list[str] = []
    for chunk_id in chunk_ids:
        chunk = chunks[chunk_id]
        body = normalize_space(str(chunk["text"]))
        if any(
            annotation.doc_id == chunk["parent_doc_id"]
            and normalize_space(annotation.fact) in body
            for annotation in evidence
        ):
            gold.append(chunk_id)
        else:
            other.append(chunk_id)
    if position == "first":
        return [*gold, *other]
    if position == "last":
        return [*other, *gold]
    midpoint = len(other) // 2
    return [*other[:midpoint], *gold, *other[midpoint:]]


def oracle_block(document: dict[str, Any], evidence: Evidence) -> str:
    metadata = [
        f"title: {document['title']}",
        f"source: {document.get('source', '')}",
        f"published_at: {document.get('published_at', '')}",
    ]
    return "\n".join([*metadata, f"evidence: {evidence.fact}"])


def find_run(path: Path, run_name: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["run_name"] == run_name]
    if not rows:
        raise ValueError(f"No retrieval run named {run_name!r} in {path}.")
    return rows[-1]


def read_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            run.setdefault(item["query_id"], {})[item["doc_id"]] = float(item["score"])
    return run


def read_chunks(path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            chunks[str(item["chunk_id"])] = item["chunk"]
    return chunks


def read_generation_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return {item["query_id"]: item for item in map(json.loads, file)}


def write_per_query(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "query_id",
        "question_type",
        "exact_match",
        "token_f1",
        "abstained",
        "response_format_valid",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def upsert_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            rows = {item["run_name"]: item for item in csv.DictReader(file)}
    rows[str(row["run_name"])] = row
    fields: list[str] = []
    for item in rows.values():
        fields.extend(field for field in item if field not in fields)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows.values())


def hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
