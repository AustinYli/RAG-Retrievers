from rag_bench.metrics import evaluate, evaluate_per_query


def test_evaluate_basic_ranked_metrics():
    qrels = {
        "q1": {"d1": 1, "d2": 1},
        "q2": {"d3": 1},
    }
    runs = {
        "q1": {"d9": 3.0, "d1": 2.0, "d2": 1.0},
        "q2": {"d3": 1.0},
    }

    metrics = evaluate(qrels, runs, cutoffs=[1, 2, 3])

    assert metrics["recall@1"] == 0.5
    assert metrics["mrr@1"] == 0.5
    assert round(metrics["recall@3"], 6) == 1.0
    assert round(metrics["map@3"], 6) == 0.791667

    per_query = evaluate_per_query(qrels, runs, cutoffs=[3])
    assert round(per_query["q1"]["ndcg@3"], 6) == 0.693426
    assert per_query["q2"]["ndcg@3"] == 1.0


def test_evaluate_missing_run_counts_as_zero():
    qrels = {"q1": {"d1": 1}}
    runs = {}

    metrics = evaluate(qrels, runs, cutoffs=[10])

    assert metrics["ndcg@10"] == 0.0
    assert metrics["recall@10"] == 0.0
    assert metrics["mrr@10"] == 0.0
    assert metrics["map@10"] == 0.0


def test_ranked_metrics_match_pytrec_eval():
    import pytrec_eval

    qrels = {
        "q1": {"d1": 1, "d2": 1},
        "q2": {"d3": 1},
        "q3": {f"d{index}": 1 for index in range(10, 25)},
        "q4": {"graded_high": 2, "graded_low": 1},
    }
    runs = {
        "q1": {"d9": 3.0, "d1": 2.0, "d2": 1.0},
        "q2": {"d3": 1.0},
        "q3": {f"d{index}": float(25 - index) for index in range(10, 20)},
        "q4": {"graded_low": 2.0, "graded_high": 1.0},
    }

    ours = evaluate(qrels, runs, cutoffs=[10])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut.10", "map_cut.10", "recall.10"}
    )
    theirs = evaluator.evaluate(runs)
    expected = {
        "ndcg@10": sum(row["ndcg_cut_10"] for row in theirs.values()) / len(qrels),
        "map@10": sum(row["map_cut_10"] for row in theirs.values()) / len(qrels),
        "recall@10": sum(row["recall_10"] for row in theirs.values()) / len(qrels),
    }

    for metric, expected_value in expected.items():
        assert round(ours[metric], 12) == round(expected_value, 12)

    assert theirs["q3"]["map_cut_10"] == 10 / 15
