from pathlib import Path

from scripts.run_track_b_stage1 import persist_chunk_artifact, rerank_checkpoint_path


def test_rerank_checkpoint_key_changes_with_every_material_dependency():
    base = {
        "corpus_sha256": "corpus-a",
        "queries_sha256": "queries-a",
        "base_run_id": "dense-a",
        "base_run_artifact_sha256": "dense-artifact-a",
        "reranker_revision": "revision-a",
        "rerank_top_k": 20,
    }
    original = rerank_checkpoint_path(Path("cache"), "friendly-name", base)

    for field in base:
        changed = {**base, field: f"{base[field]}-changed"}
        assert rerank_checkpoint_path(Path("cache"), "friendly-name", changed) != original


def test_chunk_artifact_is_content_stable_and_keyed_by_segmentation(tmp_path: Path):
    chunks = {"doc::c000": {"text": "alpha beta", "parent_doc_id": "doc"}}
    config = {"corpus_sha256": "corpus-a", "chunking_strategy": "semantic"}

    first = persist_chunk_artifact(tmp_path, chunks, config)
    second = persist_chunk_artifact(tmp_path, chunks, config)
    changed = persist_chunk_artifact(
        tmp_path,
        chunks,
        {**config, "chunking_strategy": "structural"},
    )

    assert first == second
    assert first.read_bytes() == second.read_bytes()
    assert changed != first
