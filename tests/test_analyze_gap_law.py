from scripts.analyze_gap_law import ols_slope, pearson, ranks


def test_correlation_helpers():
    assert pearson([1, 2, 3], [3, 2, 1]) == -1.0
    assert ranks([30, 10, 20]) == [3.0, 1.0, 2.0]
    assert ols_slope([1, 2, 3], [2, 4, 6]) == 2.0


def test_ranks_average_ties():
    assert ranks([10, 10, 20]) == [1.5, 1.5, 3.0]
