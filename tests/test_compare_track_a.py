from scripts.compare_track_a import dedupe_comparisons


def test_dedupe_comparisons_uses_run_pair_not_label():
    baseline = {"dataset": "scifact", "run_id": "bm25"}
    candidate = {"dataset": "scifact", "run_id": "dense-bge"}

    comparisons = dedupe_comparisons(
        [
            ("bm25_to_best_dense", baseline, candidate),
            ("bm25_to_dense_bge", baseline, candidate),
        ]
    )

    assert comparisons == [("bm25_to_best_dense", baseline, candidate)]
