from types import SimpleNamespace

import numpy as np
import pytest

from rag_bench.faithfulness import (
    claim_support_scores,
    cohen_kappa_binary,
    split_claims,
    unsupported_claim_rate,
)
from scripts.score_track_b_faithfulness import calibration_metrics


class FakeNLI:
    model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
    )

    def predict(self, pairs, batch_size, apply_softmax, show_progress_bar):
        assert apply_softmax
        assert batch_size == 32
        assert len(pairs) == 2
        return np.array([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])


class FakeMultiBlockNLI(FakeNLI):
    def predict(self, pairs, batch_size, apply_softmax, show_progress_bar):
        assert batch_size == 32
        assert len(pairs) == 4
        return np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.6, 0.3],
                [0.7, 0.2, 0.1],
            ]
        )


class FakeSingleClaimNLI(FakeNLI):
    def predict(self, pairs, batch_size, apply_softmax, show_progress_bar):
        assert len(pairs) == 1
        return np.array([[0.1, 0.8, 0.1]])


def test_claim_level_entailment_scoring():
    claims = split_claims("First claim. Second claim!")
    scores = claim_support_scores("context", "First claim. Second claim!", FakeNLI())

    assert claims == ["First claim.", "Second claim!"]
    assert scores == [0.8, 0.2]
    assert unsupported_claim_rate(scores, threshold=0.5) == 0.5


def test_claim_support_uses_best_entailing_context_block():
    scores = claim_support_scores(
        "first context\n\nsecond context",
        "First claim. Second claim!",
        FakeMultiBlockNLI(),
    )

    assert scores == [0.7, 0.6]


def test_structured_claim_can_be_scored_without_splitting_abbreviations():
    scores = claim_support_scores(
        "context",
        "The U.S. result is supported.",
        FakeSingleClaimNLI(),
        split_sentences=False,
    )

    assert scores == [0.8]


def test_binary_cohen_kappa():
    assert cohen_kappa_binary([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohen_kappa_binary([True, True, False, False], [True, False, True, False]) == 0.0


def test_calibration_rejects_an_incomplete_label_sheet(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "query_id,claim_index,human_supported\nq1,0,1\n",
        encoding="utf-8",
    )
    claims = [{"query_id": "q1", "claim_index": 0, "entailment_probability": 0.9}]

    with pytest.raises(ValueError, match="at least 50"):
        calibration_metrics(claims, labels, threshold=0.5)
