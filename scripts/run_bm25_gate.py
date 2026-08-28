from __future__ import annotations

import json
from pathlib import Path

from rag_bench.run import append_result, run_config


ANALYZERS = [
    "word-lower-v1",
    "word-lower-stop-v1",
    "word-lower-porter-v1",
    "word-lower-stop-porter-v1",
    "word-lower-bigram-v1",
]

PARAMS = [
    {"label": "rankbm25_default", "k1": 1.5, "b": 0.75, "epsilon": 0.25},
    {"label": "lucene_beir", "k1": 0.9, "b": 0.4, "epsilon": 0.25},
    {"label": "length_norm", "k1": 1.2, "b": 0.9, "epsilon": 0.25},
]


def main() -> None:
    output_path = Path("results/bm25_gate.csv")
    for analyzer in ANALYZERS:
        for params in PARAMS:
            run_name = f"scifact_bm25_gate_{analyzer}_{params['label']}"
            config = {
                "run_name": run_name,
                "dataset": "scifact",
                "split": "test",
                "data_dir": "data/beir",
                "cache_dir": "cache",
                "results_path": str(output_path),
                "artifacts_dir": "results/bm25_gate_artifacts",
                "retriever": {
                    "type": "bm25",
                    "top_k": 1000,
                    "analyzer": analyzer,
                    "k1": params["k1"],
                    "b": params["b"],
                    "epsilon": params["epsilon"],
                },
                "metrics": {"cutoffs": [1, 5, 10, 100]},
            }
            row = run_config(config, Path(f"<generated:{run_name}>"))
            append_result(output_path, row)
            print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
