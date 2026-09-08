from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate paired exact-match sample sizes from pilot discordance."
    )
    parser.add_argument(
        "--artifacts",
        nargs=3,
        required=True,
        help="Exactly three LABEL=per-query.csv artifacts in comparison order.",
    )
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--family-size", type=int, default=24)
    parser.add_argument(
        "--output", default="results/track_b_generation_sample200_power.json"
    )
    args = parser.parse_args()
    if not 0.0 < args.power < 1.0 or not 0.0 < args.alpha < 1.0:
        parser.error("--power and --alpha must be between zero and one.")
    if args.family_size <= 0:
        parser.error("--family-size must be positive.")

    artifacts: list[tuple[str, Path]] = []
    for specification in args.artifacts:
        if "=" not in specification:
            parser.error(f"Artifact must use LABEL=PATH syntax: {specification}")
        label, path_text = specification.split("=", 1)
        artifacts.append((label, Path(path_text)))
    values = {label: read_answerable_em(path) for label, path in artifacts}

    comparisons = []
    for left_index, right_index in ((0, 1), (1, 2), (0, 2)):
        left_label = artifacts[left_index][0]
        right_label = artifacts[right_index][0]
        comparisons.append(
            paired_power_summary(
                left_label,
                values[left_label],
                right_label,
                values[right_label],
                power=args.power,
                alpha=args.alpha,
                family_size=args.family_size,
            )
        )

    result = {
        "method": "mcnemar_normal_approximation_from_pilot_discordance",
        "scope": (
            "Exploratory planning estimate using observed pilot effect sizes; "
            "not a confirmatory result or guaranteed achieved power."
        ),
        "target_power": args.power,
        "two_sided_alpha": args.alpha,
        "family_size": args.family_size,
        "conservative_family_alpha": args.alpha / args.family_size,
        "comparisons": comparisons,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def paired_power_summary(
    left_label: str,
    left: dict[str, float],
    right_label: str,
    right: dict[str, float],
    *,
    power: float,
    alpha: float,
    family_size: int,
) -> dict:
    query_ids = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise ValueError(f"Query coverage differs for {left_label} and {right_label}.")
    left_only = sum(left[query_id] > right[query_id] for query_id in query_ids)
    right_only = sum(right[query_id] > left[query_id] for query_id in query_ids)
    discordant = left_only + right_only
    n = len(query_ids)
    delta = (right_only - left_only) / n
    discordance = discordant / n
    return {
        "comparison": f"{left_label}_to_{right_label}",
        "num_queries": n,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant_count": discordant,
        "discordance_rate": discordance,
        "observed_em_delta": delta,
        "estimated_n_80pct_power_alpha_0.05": required_sample_size(
            delta, discordance, alpha=alpha, power=power
        ),
        "estimated_n_80pct_power_conservative_family_alpha": required_sample_size(
            delta, discordance, alpha=alpha / family_size, power=power
        ),
    }


def required_sample_size(
    delta: float, discordance: float, *, alpha: float, power: float
) -> int | None:
    absolute_delta = abs(delta)
    if absolute_delta == 0.0:
        return None
    if not absolute_delta <= discordance <= 1.0:
        raise ValueError("Discordance must be between the absolute delta and one.")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    alternative_variance = max(discordance - absolute_delta**2, 0.0)
    numerator = (
        z_alpha * math.sqrt(discordance)
        + z_power * math.sqrt(alternative_variance)
    ) ** 2
    return math.ceil(numerator / absolute_delta**2)


def read_answerable_em(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return {
            row["query_id"]: float(row["exact_match"])
            for row in csv.DictReader(file)
            if row["question_type"] != "null_query"
        }


if __name__ == "__main__":
    main()
