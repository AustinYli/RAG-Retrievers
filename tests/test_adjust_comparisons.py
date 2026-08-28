from scripts.adjust_comparisons import holm_adjust


def test_holm_adjust_is_monotonic_in_rank_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == [0.03, 0.06, 0.06]
