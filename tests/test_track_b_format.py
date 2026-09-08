from scripts.analyze_track_b_format import analyze_format, classify_format_failure


def row(*, valid: bool, exact_match: float, token_f1: float, raw: str, null=False):
    return {
        "question_type": "null_query" if null else "inference_query",
        "response_format_valid": valid,
        "exact_match": exact_match,
        "token_f1": token_f1,
        "raw_response": raw,
    }


def test_format_analysis_does_not_treat_malformed_answers_as_automatic_zeros():
    rows = [
        row(
            valid=True,
            exact_match=0.0,
            token_f1=0.5,
            raw="Answer: wrong\nClaim: This is supported.",
        ),
        row(valid=False, exact_match=1.0, token_f1=1.0, raw="correct"),
        row(
            valid=False,
            exact_match=0.0,
            token_f1=0.0,
            raw="Answer: wrong\n\nClaim: This is supported.",
        ),
        row(
            valid=True,
            exact_match=0.0,
            token_f1=0.0,
            raw="Answer: insufficient\nClaim: Evidence is insufficient.",
            null=True,
        ),
    ]

    result = analyze_format(rows)

    assert result["answerable_exact_match_all"] == 1 / 3
    assert result["answerable_exact_match_well_formed"] == 0.0
    assert result["answerable_exact_match_malformed"] == 0.5
    assert result["answerable_malformed_wrong_count"] == 1
    assert result["fraction_of_answerable_errors_with_malformed_format"] == 0.5
    assert result["malformed_reason_counts_answerable"] == {
        "extra_or_blank_lines": 1,
        "missing_answer_line": 1,
    }


def test_format_failure_reasons_follow_parser_constraints():
    assert classify_format_failure("Claim: Supported.") == "missing_answer_line"
    assert classify_format_failure("Answer: yes") == "missing_claim_line"
    assert (
        classify_format_failure("Answer: yes\n\nClaim: Supported.")
        == "extra_or_blank_lines"
    )
    assert (
        classify_format_failure(
            "Answer: yes\nClaim: " + " ".join(["word"] * 26) + "."
        )
        == "claim_over_word_limit"
    )
    assert (
        classify_format_failure("Answer: yes\nClaim: This has no punctuation")
        == "claim_missing_terminal_punctuation"
    )
