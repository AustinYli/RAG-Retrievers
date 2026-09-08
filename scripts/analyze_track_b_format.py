from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from rag_bench.generation import MAX_CLAIM_WORDS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether response-format failures confound generation metrics."
    )
    parser.add_argument(
        "artifacts",
        nargs="+",
        help="Generation artifacts as LABEL=PATH pairs.",
    )
    parser.add_argument("--output", default="results/track_b_format_analysis.json")
    args = parser.parse_args()

    analyses = {}
    for specification in args.artifacts:
        if "=" not in specification:
            parser.error(f"Artifact must use LABEL=PATH syntax: {specification}")
        label, path_text = specification.split("=", 1)
        if not label or not path_text:
            parser.error(f"Artifact must use LABEL=PATH syntax: {specification}")
        path = Path(path_text)
        rows = read_jsonl(path)
        analyses[label] = {
            "generation_artifact": str(path),
            **analyze_format(rows),
        }

    result = {
        "interpretation": (
            "Malformed responses retain fallback predictions and are not automatically "
            "scored as zero. Conditional metrics are descriptive, not causal estimates."
        ),
        "runs": analyses,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def analyze_format(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Generation artifact is empty.")
    answerable = [row for row in rows if row["question_type"] != "null_query"]
    null_rows = [row for row in rows if row["question_type"] == "null_query"]
    if not answerable:
        raise ValueError("Generation artifact contains no answerable queries.")

    valid = [row for row in answerable if bool(row["response_format_valid"])]
    malformed = [row for row in answerable if not bool(row["response_format_valid"])]
    wrong = [row for row in answerable if float(row["exact_match"]) == 0.0]
    malformed_wrong = [row for row in malformed if float(row["exact_match"]) == 0.0]
    reasons = Counter(
        classify_format_failure(str(row["raw_response"])) for row in malformed
    )

    return {
        "num_queries": len(rows),
        "num_answerable_queries": len(answerable),
        "num_null_queries": len(null_rows),
        "response_format_valid_rate_all": rate(
            bool(row["response_format_valid"]) for row in rows
        ),
        "response_format_valid_rate_answerable": len(valid) / len(answerable),
        "response_format_valid_rate_null": (
            rate(bool(row["response_format_valid"]) for row in null_rows)
            if null_rows
            else None
        ),
        "answerable_exact_match_all": metric(answerable, "exact_match"),
        "answerable_exact_match_well_formed": metric(valid, "exact_match"),
        "answerable_exact_match_malformed": metric(malformed, "exact_match"),
        "answerable_token_f1_all": metric(answerable, "token_f1"),
        "answerable_token_f1_well_formed": metric(valid, "token_f1"),
        "answerable_token_f1_malformed": metric(malformed, "token_f1"),
        "answerable_malformed_count": len(malformed),
        "answerable_malformed_wrong_count": len(malformed_wrong),
        "answerable_wrong_count": len(wrong),
        "fraction_of_answerable_errors_with_malformed_format": (
            len(malformed_wrong) / len(wrong) if wrong else 0.0
        ),
        "malformed_reason_counts_answerable": dict(sorted(reasons.items())),
    }


def classify_format_failure(response: str) -> str:
    answer_match = re.search(r"^Answer:\s*(.+?)\s*$", response, flags=re.MULTILINE)
    claim_match = re.search(r"^Claim:\s*(.+?)\s*$", response, flags=re.MULTILINE)
    if not answer_match:
        return "missing_answer_line"
    if not claim_match:
        return "missing_claim_line"
    if not re.fullmatch(r"Answer:\s*[^\n]+\nClaim:\s*[^\n]+", response.strip()):
        return "extra_or_blank_lines"
    claim = claim_match.group(1).strip()
    if len(claim.split()) > MAX_CLAIM_WORDS:
        return "claim_over_word_limit"
    if not re.search(r"[.!?]['\")\]]*$", claim):
        return "claim_missing_terminal_punctuation"
    return "unclassified"


def metric(rows: list[dict], name: str) -> float | None:
    return mean(float(row[name]) for row in rows) if rows else None


def rate(values) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    main()
