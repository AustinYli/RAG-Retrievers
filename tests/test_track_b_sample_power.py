import pytest

from scripts.analyze_track_b_sample_power import (
    paired_power_summary,
    required_sample_size,
)


def test_power_summary_counts_paired_discordance():
    result = paired_power_summary(
        "left",
        {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 0.0},
        "right",
        {"q1": 0.0, "q2": 1.0, "q3": 1.0, "q4": 1.0},
        power=0.8,
        alpha=0.05,
        family_size=24,
    )

    assert result["left_only_correct"] == 1
    assert result["right_only_correct"] == 2
    assert result["discordant_count"] == 3
    assert result["discordance_rate"] == 0.75
    assert result["observed_em_delta"] == 0.25
    assert result["estimated_n_80pct_power_conservative_family_alpha"] > result[
        "estimated_n_80pct_power_alpha_0.05"
    ]


def test_power_estimate_rejects_impossible_discordance_and_zero_delta():
    assert required_sample_size(0.0, 0.2, alpha=0.05, power=0.8) is None
    with pytest.raises(ValueError, match="Discordance"):
        required_sample_size(0.3, 0.2, alpha=0.05, power=0.8)
