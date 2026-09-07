from scripts.check_metrics_pytrec_eval import rank_normalize


def test_rank_normalize_preserves_deterministic_tie_breaking():
    normalized = rank_normalize({"q1": {"d2": 1.0, "d1": 1.0, "d3": 0.5}})

    assert list(normalized["q1"]) == ["d1", "d2", "d3"]
    assert normalized["q1"] == {"d1": 3.0, "d2": 2.0, "d3": 1.0}
