import pytest

from scripts.analyze_track_b_determinism import (
    summarize_determinism,
    validate_determinism_metadata,
)


def metadata(repeat: int) -> dict:
    return {
        "experiment": {"model": "qwen", "seed": 13, "repeat": repeat},
        "prompt_template": "Answer from evidence.",
        "model_details": {"quantization_level": "Q4_K_M"},
        "runtime": {"platform": "macOS", "python_version": "3.13"},
    }


def test_determinism_metadata_allows_only_repeat_to_change():
    rows = [metadata(1), metadata(2), metadata(3)]

    validate_determinism_metadata(rows)

    rows[2]["experiment"]["seed"] = 29
    with pytest.raises(ValueError, match="controls"):
        validate_determinism_metadata(rows)


def test_determinism_correctness_excludes_null_queries():
    row = {
        "q1": {
            "question_type": "inference_query",
            "prediction": "right",
            "exact_match": 1,
            "token_f1": 1,
        },
        "q2": {
            "question_type": "null_query",
            "prediction": "not abstained",
            "exact_match": 0,
            "token_f1": 0,
        },
    }

    result = summarize_determinism([row, row, row])

    assert result["num_queries"] == 2
    assert result["num_answerable_queries"] == 1
    assert result["num_null_queries"] == 1
    assert result["exact_match_by_run"] == [1, 1, 1]
    assert result["exact_prediction_agreement_rate"] == 1
