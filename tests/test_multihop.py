import json
from pathlib import Path

import numpy as np

from rag_bench.multihop import (
    chunk_corpus,
    dataset_audit,
    evaluate_evidence_per_query,
    load_multihop_dataset,
)


def write_fixture(root: Path) -> None:
    corpus = [
        {
            "title": "First",
            "url": "https://example.com/first",
            "body": "one two three four five six seven eight",
        },
        {
            "title": "Second",
            "url": "https://example.com/second",
            "body": "alpha beta gamma delta epsilon zeta eta theta",
        },
    ]
    queries = [
        {
            "query": "What combines both facts?",
            "answer": "combined",
            "question_type": "inference_query",
            "evidence_list": [
                {
                    "url": "https://example.com/first",
                    "fact": "three four",
                },
                {
                    "url": "https://example.com/second",
                    "fact": "gamma delta",
                },
            ],
        },
        {
            "query": "What is missing?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    (root / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    (root / "MultiHopRAG.json").write_text(json.dumps(queries), encoding="utf-8")


def test_loader_validates_schema_and_audits_counts(tmp_path: Path):
    write_fixture(tmp_path)

    dataset = load_multihop_dataset(tmp_path, download=False)
    audit = dataset_audit(dataset)

    assert len(dataset.corpus) == 2
    assert dataset.answerable_query_ids == ["mhq-0000"]
    assert dataset.null_query_ids == ["mhq-0001"]
    assert audit["question_type_counts"] == {"inference_query": 1, "null_query": 1}
    assert audit["mean_unique_evidence_documents_answerable"] == 2


def test_chunk_metrics_separate_document_recall_from_fact_coverage(tmp_path: Path):
    write_fixture(tmp_path)
    dataset = load_multihop_dataset(tmp_path, download=False)
    chunks, chunk_to_doc = chunk_corpus(dataset.corpus, chunk_words=4, overlap_words=0)
    first_doc, second_doc = list(dataset.corpus)
    run = {
        "mhq-0000": {
            f"{first_doc}::c001": 3.0,
            f"{first_doc}::c000": 2.0,
            f"{second_doc}::c000": 1.0,
        }
    }

    metrics = evaluate_evidence_per_query(
        dataset.evidence, run, chunks, chunk_to_doc, cutoffs=[1, 2, 3]
    )["mhq-0000"]

    assert metrics["evidence_doc_recall@1"] == 0.5
    assert metrics["evidence_recall@1"] == 0.0
    assert metrics["evidence_recall@2"] == 0.5
    assert metrics["context_sufficiency@3"] == 1.0


def test_chunks_expose_query_relevant_article_metadata(tmp_path: Path):
    write_fixture(tmp_path)
    dataset = load_multihop_dataset(tmp_path, download=False)
    chunks, _ = chunk_corpus(dataset.corpus, chunk_words=4, overlap_words=0)

    first_chunk = next(iter(chunks.values()))

    assert first_chunk["title"] == "title: First"
    assert first_chunk["article_title"] == "First"


def test_structural_chunks_preserve_paragraph_boundaries_when_they_fit():
    corpus = {
        "doc": {
            "title": "Story",
            "text": "one two three.\n\nfour five six.\n\nseven eight nine.",
        }
    }

    chunks, _ = chunk_corpus(
        corpus,
        chunk_words=6,
        overlap_words=0,
        strategy="structural",
    )

    assert [chunk["text"] for chunk in chunks.values()] == [
        "one two three. four five six.",
        "seven eight nine.",
    ]
    assert all(chunk["chunking_strategy"] == "structural" for chunk in chunks.values())


class FakeSemanticEncoder:
    def encode(self, sentences, **kwargs):
        vectors = {
            "Alpha stays here.": [1.0, 0.0],
            "Alpha remains nearby.": [1.0, 0.0],
            "Beta changes topic.": [0.0, 1.0],
            "Beta stays there.": [0.0, 1.0],
        }
        return np.asarray([vectors[sentence] for sentence in sentences])


def test_semantic_chunks_break_at_low_similarity_boundary():
    corpus = {
        "doc": {
            "title": "Story",
            "text": (
                "Alpha stays here. Alpha remains nearby. "
                "Beta changes topic. Beta stays there."
            ),
        }
    }

    chunks, _ = chunk_corpus(
        corpus,
        chunk_words=20,
        overlap_words=0,
        strategy="semantic",
        semantic_encoder=FakeSemanticEncoder(),
        semantic_break_percentile=15,
        semantic_min_chunk_words=3,
    )

    assert [chunk["text"] for chunk in chunks.values()] == [
        "Alpha stays here. Alpha remains nearby.",
        "Beta changes topic. Beta stays there.",
    ]


def test_nonfixed_chunking_rejects_overlap():
    corpus = {"doc": {"title": "Story", "text": "one two three"}}

    try:
        chunk_corpus(corpus, strategy="structural", overlap_words=1)
    except ValueError as error:
        assert "Overlap" in str(error)
    else:
        raise AssertionError("Expected structural overlap to be rejected.")
