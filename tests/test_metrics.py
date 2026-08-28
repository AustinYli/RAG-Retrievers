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


def test_ndcg_matches_pytrec_eval():
    import pytrec_eval

    qrels = {
        "q1": {"d1": 1, "d2": 1},
        "q2": {"d3": 1},
    }
    runs = {
        "q1": {"d9": 3.0, "d1": 2.0, "d2": 1.0},
        "q2": {"d3": 1.0},
    }

    ours = evaluate(qrels, runs, cutoffs=[10])["ndcg@10"]
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    theirs = sum(row["ndcg_cut_10"] for row in evaluator.evaluate(runs).values()) / len(qrels)

    assert round(ours, 12) == round(theirs, 12)
