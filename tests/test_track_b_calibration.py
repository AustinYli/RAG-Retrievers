from scripts.sample_faithfulness_calibration import sample_claims


def test_calibration_sample_is_stable_and_input_order_independent():
    claims = [
        {"query_id": "q1", "claim_index": 0},
        {"query_id": "q2", "claim_index": 0},
        {"query_id": "q3", "claim_index": 0},
    ]

    forward = sample_claims(claims, samples=2, seed=13)
    backward = sample_claims(list(reversed(claims)), samples=2, seed=13)

    assert forward == backward
    assert len(forward) == 2
