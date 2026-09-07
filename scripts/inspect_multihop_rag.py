from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_bench.multihop import (
    chunk_corpus,
    dataset_audit,
    load_multihop_dataset,
    normalize_space,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the downloaded MultiHop-RAG files.")
    parser.add_argument("--data-dir", default="data/multihop_rag")
    parser.add_argument("--output", default="results/track_b_dataset_audit.json")
    parser.add_argument("--chunk-words", type=int, default=256)
    parser.add_argument("--overlap-words", type=int, default=64)
    args = parser.parse_args()

    dataset = load_multihop_dataset(args.data_dir)
    chunks, _ = chunk_corpus(
        dataset.corpus,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    chunks_by_document: dict[str, list[str]] = {}
    for chunk in chunks.values():
        chunks_by_document.setdefault(chunk["parent_doc_id"], []).append(
            normalize_space(chunk["text"])
        )
    covered_annotations = sum(
        any(
            normalize_space(annotation.fact) in chunk
            for chunk in chunks_by_document[annotation.doc_id]
        )
        for annotations in dataset.evidence.values()
        for annotation in annotations
    )
    audit = {
        **dataset_audit(dataset),
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "chunk_unit": "whitespace_word",
        "indexed_metadata": ["title", "source", "published_at"],
        "chunk_count": len(chunks),
        "chunks_per_document_mean": len(chunks) / len(dataset.corpus),
        "evidence_annotations_covered_by_some_chunk": covered_annotations,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
