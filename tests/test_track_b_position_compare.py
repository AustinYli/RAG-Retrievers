import pytest

from scripts.compare_track_b_positions import validate_position_controls


def row(position: str) -> dict[str, str]:
    return {
        "mode": "retrieved",
        "evidence_position": position,
        "retriever_run_id": "retriever",
        "model": "qwen",
        "prompt_sha256": "prompt",
        "context_packing": "balanced",
    }


def test_position_comparison_allows_only_order_to_change():
    rows = [row("first"), row("middle"), row("last")]

    validate_position_controls(rows)

    rows[1]["model"] = "other"
    with pytest.raises(ValueError, match="model"):
        validate_position_controls(rows)


def test_position_comparison_requires_declared_order():
    with pytest.raises(ValueError, match="first,middle,last"):
        validate_position_controls([row("last"), row("middle"), row("first")])
