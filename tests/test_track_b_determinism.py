import pytest

from scripts.analyze_track_b_determinism import validate_determinism_metadata


def metadata(repeat: int) -> dict:
    return {
        "experiment": {"model": "qwen", "seed": 13, "repeat": repeat},
        "prompt_template": "Answer from evidence.",
        "model_details": {"quantization_level": "Q4_K_M"},
        "runtime": {"platform": "macOS", "python_version": "3.13"},
    }


def test_determinism_metadata_allows_only_repeat_to_change():
    rows = [metadata(1), metadata(2), metadata(3)]

    validate_determinism_metadata(rows)

    rows[2]["experiment"]["seed"] = 29
    with pytest.raises(ValueError, match="controls"):
        validate_determinism_metadata(rows)
