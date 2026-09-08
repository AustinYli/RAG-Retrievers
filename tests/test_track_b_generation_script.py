from types import SimpleNamespace

from rag_bench.multihop import Evidence
from scripts.run_track_b_generation import (
    CLOSED_BOOK_PROMPT_TEMPLATE,
    balanced_word_allocations,
    build_context,
    duration_components_are_valid,
    greedy_context,
    position_evidence_chunks,
)


def test_closed_book_mode_has_no_context_and_explicitly_allows_model_knowledge():
    context, context_ids = build_context(
        mode="closed_book",
        query_id="q1",
        dataset=SimpleNamespace(),
        chunks={},
        ranked_runs={},
        top_k=5,
        max_context_words=1280,
        evidence_position="natural",
    )

    assert context == ""
    assert context_ids == []
    assert "own knowledge" in CLOSED_BOOK_PROMPT_TEMPLATE
    assert "Evidence:" not in CLOSED_BOOK_PROMPT_TEMPLATE


def test_position_variants_hold_the_retrieved_set_fixed():
    chunks = {
        "gold": {"parent_doc_id": "d1", "text": "the supported fact"},
        "noise-1": {"parent_doc_id": "d2", "text": "noise"},
        "noise-2": {"parent_doc_id": "d3", "text": "more noise"},
    }
    ranked = ["noise-1", "gold", "noise-2"]
    evidence = (Evidence(doc_id="d1", fact="supported fact"),)

    assert position_evidence_chunks(ranked, evidence, chunks, "first") == [
        "gold",
        "noise-1",
        "noise-2",
    ]
    assert position_evidence_chunks(ranked, evidence, chunks, "middle") == [
        "noise-1",
        "gold",
        "noise-2",
    ]
    assert position_evidence_chunks(ranked, evidence, chunks, "last") == [
        "noise-1",
        "noise-2",
        "gold",
    ]
    assert set(position_evidence_chunks(ranked, evidence, chunks, "last")) == set(ranked)


def test_retrieved_context_respects_total_rendered_word_budget():
    dataset = SimpleNamespace(evidence={"q1": ()})
    chunks = {
        "c1": {"title": "title one", "text": "one two three"},
        "c2": {"title": "title two", "text": "four five six"},
    }
    context, context_ids = build_context(
        mode="retrieved",
        query_id="q1",
        dataset=dataset,
        chunks=chunks,
        ranked_runs={"q1": {"c1": 2.0, "c2": 1.0}},
        top_k=2,
        max_context_words=8,
        evidence_position="natural",
    )

    assert context_ids == ["c1", "c2"]
    assert "one" in context
    assert "four" in context
    assert len(context.split()) <= 8


def test_balanced_budget_does_not_depend_on_display_order():
    forward = balanced_word_allocations([("a", 10), ("b", 10)], 5)
    reversed_order = balanced_word_allocations([("b", 10), ("a", 10)], 5)

    assert forward == reversed_order == {"a": 3, "b": 2}


def test_greedy_context_fills_budget_in_rank_order_and_truncates_last_chunk():
    chunks = {
        "c1": {"title": "source one", "text": "a b c"},
        "c2": {"title": "source two", "text": "d e f g"},
        "c3": {"title": "source three", "text": "h i"},
    }

    context, context_ids = greedy_context(chunks, ["c1", "c2", "c3"], 12)

    assert context_ids == ["c1", "c2"]
    assert context.split() == [
        "[1]",
        "source",
        "one",
        "a",
        "b",
        "c",
        "[2]",
        "source",
        "two",
        "d",
        "e",
        "f",
    ]


def test_impossible_ollama_component_durations_are_rejected():
    valid = {
        "total_duration_ns": 100,
        "load_duration_ns": 10,
        "prompt_eval_duration_ns": 30,
        "eval_duration_ns": 50,
    }

    assert duration_components_are_valid(valid)
    assert not duration_components_are_valid({**valid, "eval_duration_ns": 70})
