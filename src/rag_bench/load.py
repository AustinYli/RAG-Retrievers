from __future__ import annotations

from pathlib import Path
from typing import Any

from beir import util
from beir.datasets.data_loader import GenericDataLoader


BEIR_DATASETS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


def load_beir_dataset(
    dataset: str,
    split: str = "test",
    data_dir: str | Path = "data/beir",
    max_corpus_docs: int | None = None,
    max_queries: int | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """Download if necessary and load a BEIR dataset."""
    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    dataset_path = data_root / dataset
    if not dataset_path.exists():
        url = BEIR_DATASETS_URL.format(dataset=dataset)
        downloaded_path = Path(util.download_and_unzip(url, str(data_root)))
        if downloaded_path != dataset_path and not dataset_path.exists():
            dataset_path = downloaded_path

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=split)

    if max_corpus_docs is not None:
        allowed_doc_ids = set(list(corpus.keys())[:max_corpus_docs])
        corpus = {doc_id: doc for doc_id, doc in corpus.items() if doc_id in allowed_doc_ids}
        qrels = {
            query_id: {doc_id: score for doc_id, score in rels.items() if doc_id in allowed_doc_ids}
            for query_id, rels in qrels.items()
        }
        qrels = {query_id: rels for query_id, rels in qrels.items() if rels}

    if max_queries is not None:
        allowed_query_ids = set(list(queries.keys())[:max_queries])
        queries = {query_id: text for query_id, text in queries.items() if query_id in allowed_query_ids}
        qrels = {query_id: rels for query_id, rels in qrels.items() if query_id in allowed_query_ids}

    queries = {query_id: text for query_id, text in queries.items() if query_id in qrels}
    return corpus, queries, qrels


def corpus_text(doc: dict[str, Any]) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title}\n{text}".strip() if title else text
