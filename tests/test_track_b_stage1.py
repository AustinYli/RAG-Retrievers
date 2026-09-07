from pathlib import Path

from argparse import Namespace

from scripts.run_track_b_stage1 import (
    FROZEN_BM25_WEIGHT,
    FROZEN_DENSE_WEIGHT,
    FROZEN_RRF_K,
    make_run_name,
    persist_chunk_artifact,
    rerank_checkpoint_path,
    stage_config,
)


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


def test_hybrid_uses_preregistered_equal_weight_rrf():
    args = Namespace(
        data_dir="data/multihop_rag",
        chunk_words=256,
        overlap_words=64,
        chunking_strategy="fixed",
        retrieval_top_k=100,
        batch_size=64,
        max_seq_length=512,
        rerank_top_k=20,
        rerank_batch_size=16,
        rerank_query_group_size=16,
        semantic_break_percentile=15.0,
        semantic_min_chunk_words=64,
        max_queries=None,
    )

    config = stage_config("hybrid", args)

    assert config["rrf_k"] == FROZEN_RRF_K == 60
    assert config["bm25_weight"] == FROZEN_BM25_WEIGHT == 1.0
    assert config["dense_weight"] == FROZEN_DENSE_WEIGHT == 1.0
    assert make_run_name("hybrid", args) == (
        "multihop_hybrid_bm25_bge_rrf60_w256_o64"
    )
