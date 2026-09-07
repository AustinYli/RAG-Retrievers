from pathlib import Path

import pytest

from rag_bench.compare import compare, compare_metric_values
from rag_bench.stats import metric_delta


def test_compare_reports_paired_bootstrap_delta(tmp_path: Path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    baseline.write_text("query_id,ndcg@10\nq1,0.0\nq2,1.0\n", encoding="utf-8")
    candidate.write_text("query_id,ndcg@10\nq1,1.0\nq2,1.0\n", encoding="utf-8")

    row = compare(baseline, candidate, metric="ndcg@10", samples=100, seed=1)

    assert row["baseline_mean"] == 0.5
    assert row["candidate_mean"] == 1.0
    assert row["delta"] == 0.5
    assert row["ci_low"] <= row["delta"] <= row["ci_high"]
    assert 0.0 <= row["p_two_sided"] <= 1.0
    assert row["baseline_run_id"] == "baseline"
    assert row["candidate_run_id"] == "candidate"


def test_metric_delta_rejects_missing_query_ids():
    try:
        metric_delta({"q1": 1.0}, {}, ["q1"])
    except KeyError:
        pass
    else:
        raise AssertionError("metric_delta should reject missing paired query IDs")


def test_compare_metric_values_rejects_empty_or_mismatched_pairs():
    with pytest.raises(ValueError, match="at least one query"):
        compare_metric_values({}, {}, metric="exact_match")
    with pytest.raises(ValueError, match="identical query IDs"):
        compare_metric_values({"q1": 1.0}, {"q2": 1.0}, metric="exact_match")
