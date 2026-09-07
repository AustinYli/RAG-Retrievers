import csv

import pytest

from rag_bench.rerank import rerank_at_depths_with_latencies
from scripts.run_rerank_sweep import annotate_latency_method


class DummyCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append((pairs, batch_size, show_progress_bar))
        return [self.scores[document] for _, document in pairs]


def test_multi_depth_reranking_scores_each_candidate_once():
    model = DummyCrossEncoder({"doc 1": 0.1, "doc 2": 0.9, "doc 3": 0.8})
    queries = {"q1": "query"}
    corpus = {
        "d1": {"title": "", "text": "doc 1"},
        "d2": {"title": "", "text": "doc 2"},
        "d3": {"title": "", "text": "doc 3"},
        "d4": {"title": "", "text": "doc 4"},
    }
    base_runs = {"q1": {"d1": 4.0, "d2": 3.0, "d3": 2.0, "d4": 1.0}}

    runs, latencies = rerank_at_depths_with_latencies(
        queries,
        corpus,
        base_runs,
        model,
        rerank_top_k_values=[2, 3],
        keep_top_k=4,
        batch_size=2,
    )

    assert list(runs[2]["q1"]) == ["d2", "d1", "d3", "d4"]
    assert list(runs[3]["q1"]) == ["d2", "d3", "d1", "d4"]
    assert [[document for _, document in call[0]] for call in model.calls] == [
        ["doc 1", "doc 2"],
        ["doc 3"],
    ]
    assert len(latencies[2]) == len(latencies[3]) == 1


def test_reranking_rejects_missing_model_scores():
    model = DummyCrossEncoder({"doc 1": 0.1})
    model.predict = lambda pairs, batch_size, show_progress_bar: []

    with pytest.raises(ValueError):
        rerank_at_depths_with_latencies(
            {"q1": "query"},
            {"d1": {"title": "", "text": "doc 1"}},
            {"q1": {"d1": 1.0}},
            model,
            rerank_top_k_values=[1],
        )


def test_annotate_latency_method_backfills_existing_rows(tmp_path):
    result_path = tmp_path / "rerank.csv"
    result_path.write_text(
        "run_name,query_latency_p50_ms\ntoy,12.3\n", encoding="utf-8"
    )

    annotate_latency_method(result_path)

    with result_path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["query_latency_method"] == "component_percentile_sum"
