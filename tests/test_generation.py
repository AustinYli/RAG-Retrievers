import pytest

from rag_bench.generation import (
    abstention_metrics,
    exact_match,
    is_abstention,
    normalize_answer,
    parse_answer_and_claim,
    prompt_hash,
    stable_query_sample,
    token_f1,
)


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("The, Eiffel Tower!", "eiffel tower"),
        ("An Apple", "apple"),
        ("A cat", "cat"),
        ("THE", ""),
        ("U.S.A.", "usa"),
        ("rock-and-roll", "rockandroll"),
        ("don't", "dont"),
        ("alpha_beta", "alphabeta"),
        ("C++", "c"),
        ("Paris\n\tFrance", "paris france"),
        ("  2024  ", "2024"),
        ("", ""),
    ],
)
def test_squad_style_normalization_cases(raw, normalized):
    assert normalize_answer(raw) == normalized


def test_squad_style_scores():
    assert exact_match("An Apple", "apple") == 1.0
    assert token_f1("red blue blue", "blue blue green") == 2 / 3
    assert token_f1("", "") == 1.0


def test_structured_response_parser_separates_answer_from_faithfulness_claim():
    answer, claim, valid = parse_answer_and_claim(
        "Answer: Sam Bankman-Fried\nClaim: Sam Bankman-Fried faced the trial."
    )

    assert answer == "Sam Bankman-Fried"
    assert claim == "Sam Bankman-Fried faced the trial."
    assert valid


def test_structured_response_parser_preserves_malformed_output_as_fallback():
    answer, claim, valid = parse_answer_and_claim("Sam Bankman-Fried")

    assert answer == claim == "Sam Bankman-Fried"
    assert not valid


def test_structured_response_parser_scores_fields_but_flags_extra_text():
    answer, claim, valid = parse_answer_and_claim(
        "Answer: Paris\nClaim: Paris is the capital.\nExtra explanation"
    )

    assert answer == "Paris"
    assert claim == "Paris is the capital."
    assert not valid


@pytest.mark.parametrize(
    "claim",
    [
        "This claim was truncated before completion",
        " ".join(["word"] * 26) + ".",
    ],
)
def test_structured_response_parser_rejects_incomplete_or_long_claims(claim):
    answer, parsed_claim, valid = parse_answer_and_claim(
        f"Answer: Yes\nClaim: {claim}"
    )

    assert answer == "Yes"
    assert parsed_claim == claim
    assert not valid


def test_abstention_reports_both_sides():
    predictions = {
        "null-1": "I don't know from the supplied context.",
        "null-2": "A confident guess",
        "answerable-1": "Insufficient evidence.",
        "answerable-2": "Sam Bankman-Fried",
    }
    types = {
        "null-1": "null_query",
        "null-2": "null_query",
        "answerable-1": "inference_query",
        "answerable-2": "inference_query",
    }

    metrics = abstention_metrics(predictions, types)

    assert is_abstention(predictions["null-1"])
    assert metrics["null_abstention_rate"] == 0.5
    assert metrics["answerable_false_abstention_rate"] == 0.5
    assert metrics["abstention_separation"] == 0.0


def test_abstention_does_not_invent_a_rate_for_an_absent_subset():
    metrics = abstention_metrics(
        {"answerable": "Paris"}, {"answerable": "inference_query"}
    )

    assert metrics["null_abstention_rate"] is None
    assert metrics["answerable_false_abstention_rate"] == 0.0
    assert metrics["abstention_separation"] is None


@pytest.mark.parametrize(
    "answer",
    [
        "The evidence does not provide an answer.",
        "There is no supporting evidence for that conclusion.",
        "That detail is not stated in the context.",
    ],
)
def test_abstention_detector_covers_frozen_context_refusals(answer):
    assert is_abstention(answer)


def test_prompt_hash_is_stable_and_sensitive_to_edits():
    assert prompt_hash("prompt A") == prompt_hash("prompt A")
    assert prompt_hash("prompt A") != prompt_hash("prompt B")


def test_stable_query_sample_is_order_independent_but_preserves_source_order():
    query_ids = ["q1", "q2", "q3", "q4"]
    selected = stable_query_sample(query_ids, size=2, seed=13)
    reversed_selected = stable_query_sample(list(reversed(query_ids)), size=2, seed=13)

    assert set(selected) == set(reversed_selected)
    assert selected == [query_id for query_id in query_ids if query_id in set(selected)]


def test_stable_query_sample_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="Sample size"):
        stable_query_sample(["q1"], size=2, seed=13)
