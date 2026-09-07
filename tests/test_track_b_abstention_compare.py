import json

import pytest

from rag_bench.generation import ABSTENTION_INSTRUCTION_ON
from scripts.compare_track_b_abstention import (
    validate_abstention_controls,
    validate_prompt_difference,
)


def test_abstention_comparison_allows_only_the_instruction_to_change():
    off = {"abstention_instruction": "off", "model": "qwen"}
    on = {"abstention_instruction": "on", "model": "qwen"}

    validate_abstention_controls(off, on)

    on["model"] = "other"
    with pytest.raises(ValueError, match="model"):
        validate_abstention_controls(off, on)


def test_abstention_comparison_checks_exact_prompt_difference(tmp_path):
    off_path = tmp_path / "off.json"
    on_path = tmp_path / "on.json"
    off_path.write_text(json.dumps({"prompt_template": "before\n\nafter"}))
    on_path.write_text(
        json.dumps(
            {"prompt_template": f"before\n{ABSTENTION_INSTRUCTION_ON}\nafter"}
        )
    )

    validate_prompt_difference(
        {"metadata_path": str(off_path)}, {"metadata_path": str(on_path)}
    )
