from types import SimpleNamespace

from rag_bench.multihop import Evidence
from scripts.analyze_track_b_generation import attribute_failures


def test_failure_attribution_uses_the_actual_saved_context():
    dataset = SimpleNamespace(
        evidence={
            "retrieval-failed": (Evidence("d1", "gold fact one"),),
            "generation-failed": (Evidence("d2", "gold fact two"),),
            "correct-from-memory": (Evidence("d3", "gold fact three"),),
        }
    )
    rows = {
        "retrieval-failed": {"context": "noise", "exact_match": 0.0},
        "generation-failed": {"context": "contains gold fact two", "exact_match": 0.0},
        "correct-from-memory": {"context": "other noise", "exact_match": 1.0},
    }

    result = attribute_failures(dataset, rows)

    assert result["wrong_missing_complete_gold_evidence"] == 1
    assert result["wrong_despite_complete_gold_evidence"] == 1
    assert result["correct_without_sufficient_context"] == 1
    assert result["actual_context_sufficiency_count"] == 1
