import pytest

from scripts.compare_track_b_hybrid import (
    evaluate_prediction,
    validate_hybrid_config,
)


def test_preregistered_prediction_requires_largest_fusion_benefit():
    prior = [0.01 * index for index in range(9)]

    assert evaluate_prediction(0.081, prior)["status"] == "confirmed"
    assert evaluate_prediction(0.08, prior)["status"] == "not_confirmed"


def test_hybrid_config_is_frozen():
    valid = {"rrf_k": "60", "bm25_weight": "1.0", "dense_weight": "1.0"}
    validate_hybrid_config(valid)

    with pytest.raises(ValueError, match="preregistered"):
        validate_hybrid_config({**valid, "rrf_k": "2"})
