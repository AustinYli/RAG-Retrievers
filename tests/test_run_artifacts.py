from pathlib import Path

from rag_bench.run import write_artifacts


def test_write_artifacts_persists_ranked_run_and_per_query_metrics(tmp_path: Path):
    paths = write_artifacts(
        artifacts_dir=tmp_path,
        run_id="test-run",
        run={"q1": {"d2": 0.5, "d1": 1.0}},
        per_query={"q1": {"ndcg@10": 1.0}},
        config={"dataset": "toy"},
        row_metadata={"run_name": "toy"},
    )

    run_text = Path(paths["run_artifact_path"]).read_text(encoding="utf-8")
    per_query_text = Path(paths["per_query_metrics_path"]).read_text(encoding="utf-8")
    metadata_text = Path(paths["metadata_path"]).read_text(encoding="utf-8")

    assert '"doc_id": "d1"' in run_text.splitlines()[0]
    assert "query_id,ndcg@10" in per_query_text
    assert '"dataset": "toy"' in metadata_text
