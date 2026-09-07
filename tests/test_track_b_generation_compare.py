import pytest

from scripts.compare_track_b_generation import validate_generation_controls


def run_row(retriever: str) -> dict[str, str]:
    return {
        "mode": "retrieved",
        "retriever_run_name": retriever,
        "model": "qwen",
        "model_digest": "digest",
        "prompt_sha256": "prompt",
        "answer_normalizer_version": "normalizer-v1",
        "response_parser_version": "parser-v1",
        "context_builder_version": "context-v1",
        "evaluation_partition_version": "evaluation-v1",
        "prompt_development_query_count": "20",
        "prompt_development_query_ids_sha256": "development",
        "abstention_patterns_sha256": "patterns",
        "temperature": "0.0",
        "seed": "13",
        "context_window": "8192",
        "max_new_tokens": "64",
        "retrieved_top_k": "5",
        "max_context_words": "1280",
        "context_packing": "balanced",
        "evidence_position": "natural",
        "abstention_instruction": "on",
        "repeat": "1",
        "query_scope": "all_queries",
        "selected_query_ids_sha256": "selected",
        "evaluation_sample_size": "",
        "evaluation_sample_seed": "",
        "evaluation_sampler_version": "",
        "queries_sha256": "queries",
        "corpus_sha256": "corpus",
        "platform": "macOS",
        "python_version": "3.13",
    }


def test_generation_comparison_allows_only_retriever_to_change():
    rows = [run_row("bm25"), run_row("dense"), run_row("reranker")]

    validate_generation_controls(rows)

    rows[1]["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="prompt_sha256"):
        validate_generation_controls(rows)


def test_generation_comparison_rejects_mixed_context_packing():
    rows = [run_row("bm25"), run_row("dense"), run_row("reranker")]
    rows[1]["context_packing"] = "greedy"

    with pytest.raises(ValueError, match="context_packing"):
        validate_generation_controls(rows)
