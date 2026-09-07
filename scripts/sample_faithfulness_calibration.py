from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SAMPLER_VERSION = "sha256-seed-query-claim-index-v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample generated claims for blinded human support labels."
    )
    parser.add_argument("--claim-artifact", required=True)
    parser.add_argument("--output", default="results/track_b_faithfulness_calibration.csv")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    with Path(args.claim_artifact).open("r", encoding="utf-8") as file:
        claims = list(map(json.loads, file))
    if args.samples <= 0:
        raise ValueError("Calibration sample count must be positive.")
    if args.samples > len(claims):
        raise ValueError("Requested more calibration samples than available claims.")
    sampled = sample_claims(claims, samples=args.samples, seed=args.seed)
    rows = [
        {
            "query_id": row["query_id"],
            "claim_index": row["claim_index"],
            "sample_seed": args.seed,
            "sample_size": args.samples,
            "sampler_version": SAMPLER_VERSION,
            "question": row.get("question", ""),
            "claim": row["claim"],
            "context": row["context"],
            "human_supported": "",
        }
        for row in sampled
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_claims(claims: list[dict], samples: int, seed: int) -> list[dict]:
    def rank(row: dict) -> str:
        identity = f"{seed}\0{row['query_id']}\0{row['claim_index']}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return sorted(claims, key=rank)[:samples]


if __name__ == "__main__":
    main()
