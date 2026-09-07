import csv
import json

from scripts.rescore_results import rescore_file


def test_rescore_file_rewrites_aggregate_and_per_query_metrics(tmp_path, monkeypatch):
    run_path = tmp_path / "run.jsonl"
    run_path.write_text(
        "\n".join(
            json.dumps({"query_id": "q1", "doc_id": f"d{index}", "score": 20 - index})
            for index in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    per_query_path = tmp_path / "per_query.csv"
    result_path = tmp_path / "results.csv"
    fieldnames = [
        "dataset",
        "split",
        "run_artifact_path",
        "per_query_metrics_path",
        "map@10",
    ]
    with result_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "fixture",
                "split": "test",
                "run_artifact_path": run_path,
                "per_query_metrics_path": per_query_path,
                "map@10": "1.0",
            }
        )

    monkeypatch.setattr(
        "scripts.rescore_results.load_beir_dataset",
        lambda **_: ({}, {}, {"q1": {f"d{index}": 1 for index in range(15)}}),
    )

    assert rescore_file(result_path, tmp_path) == (1, 0)
    with result_path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert float(row["map@10"]) == 0.666667
    assert "0.6666666666666666" in per_query_path.read_text(encoding="utf-8")
