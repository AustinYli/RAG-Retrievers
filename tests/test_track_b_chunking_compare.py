import json
from types import SimpleNamespace

import pytest

from rag_bench.multihop import Evidence
from scripts.compare_track_b_chunking import (
    chunking_label,
    evaluate_generation_artifact,
    validate_retrieval_controls,
)


def test_chunk_budget_metrics_use_rendered_context(tmp_path):
    artifact = tmp_path / "run.jsonl"
    artifact.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "context": "prefix alpha fact suffix",
                "context_ids": ["d1::c000", "d2::c000"],
                "exact_match": 1,
                "token_f1": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        question_types={"q1": "inference_query"},
        evidence={
            "q1": (
                Evidence(doc_id="d1", fact="alpha fact"),
                Evidence(doc_id="d2", fact="missing fact"),
            )
        },
    )

    values, context_counts = evaluate_generation_artifact(artifact, dataset)

    assert values["evidence_recall_at_budget"] == {"q1": 0.5}
    assert values["context_sufficiency_at_budget"] == {"q1": 0.0}
    assert context_counts == [2]


def retrieval_row(strategy="fixed", overlap="64"):
    return {
        "retriever_type": "bm25",
        "chunking_strategy": strategy,
        "overlap_words": overlap,
        "chunk_words": "256",
        "bm25_analyzer": "porter",
    }


def test_chunking_labels_distinguish_fixed_overlap():
    assert chunking_label(retrieval_row()) == "fixed_overlap_64"
    assert chunking_label(retrieval_row("structural", "0")) == "structural"


def test_chunking_comparison_rejects_retriever_changes():
    rows = [
        retrieval_row(),
        retrieval_row("fixed", "0"),
        retrieval_row("structural", "0"),
        retrieval_row("semantic", "0"),
    ]
    validate_retrieval_controls(rows)

    rows[1]["bm25_analyzer"] = "other"
    with pytest.raises(ValueError, match="bm25_analyzer"):
        validate_retrieval_controls(rows)
