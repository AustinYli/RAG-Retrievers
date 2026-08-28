from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Holm-Bonferroni adjusted p-values to a comparisons CSV.")
    parser.add_argument("--input", required=True, help="Comparison CSV with a p_two_sided column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()
    adjust_comparisons(Path(args.input), Path(args.output))


def adjust_comparisons(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    tested = [
        (index, float(row["p_two_sided"]))
        for index, row in enumerate(rows)
        if row.get("p_two_sided") not in (None, "")
    ]
    adjusted = holm_adjust([p_value for _, p_value in tested])
    for (index, _), adjusted_p in zip(tested, adjusted, strict=True):
        rows[index]["p_holm"] = f"{adjusted_p:.6f}"

    if "p_holm" not in fieldnames:
        fieldnames.append("p_holm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted_ordered: list[tuple[int, float]] = []
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_ordered.append((original_index, running_max))
    result = [1.0] * count
    for original_index, adjusted in adjusted_ordered:
        result[original_index] = adjusted
    return result


if __name__ == "__main__":
    main()
