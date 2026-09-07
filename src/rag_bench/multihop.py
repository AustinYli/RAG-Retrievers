from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Protocol, Sequence

import numpy as np


DATASET_URL = (
    "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/"
    "MultiHopRAG.json"
)
CORPUS_URL = (
    "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json"
)
ANSWERABLE_TYPES = {"comparison_query", "inference_query", "temporal_query"}
NULL_TYPE = "null_query"
CHUNKING_STRATEGIES = {"fixed", "structural", "semantic"}


class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    fact: str


@dataclass(frozen=True)
class MultiHopDataset:
    corpus: dict[str, dict[str, Any]]
    queries: dict[str, str]
    answers: dict[str, str]
    question_types: dict[str, str]
    evidence: dict[str, tuple[Evidence, ...]]
    source_hashes: dict[str, str]

    @property
    def answerable_query_ids(self) -> list[str]:
        return [
            query_id
            for query_id, question_type in self.question_types.items()
            if question_type != NULL_TYPE
        ]

    @property
    def null_query_ids(self) -> list[str]:
        return [
            query_id
            for query_id, question_type in self.question_types.items()
            if question_type == NULL_TYPE
        ]


def load_multihop_dataset(
    data_dir: str | Path = "data/multihop_rag",
    download: bool = True,
    max_queries: int | None = None,
) -> MultiHopDataset:
    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    query_path = data_root / "MultiHopRAG.json"
    corpus_path = data_root / "corpus.json"
    if download:
        _download_if_missing(DATASET_URL, query_path)
        _download_if_missing(CORPUS_URL, corpus_path)
    if not query_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            f"Expected MultiHop-RAG files at {query_path} and {corpus_path}."
        )

    raw_queries = _read_json_list(query_path)
    raw_corpus = _read_json_list(corpus_path)
    if max_queries is not None:
        raw_queries = raw_queries[:max_queries]

    corpus: dict[str, dict[str, Any]] = {}
    url_to_doc_id: dict[str, str] = {}
    for item in raw_corpus:
        _require_fields(item, {"title", "url", "body"}, "corpus record")
        url = str(item["url"])
        doc_id = document_id(url)
        if url in url_to_doc_id or doc_id in corpus:
            raise ValueError(f"Duplicate corpus URL or document ID: {url}")
        url_to_doc_id[url] = doc_id
        corpus[doc_id] = {
            "title": str(item["title"]),
            "text": str(item["body"]),
            **{key: value for key, value in item.items() if key != "body"},
        }

    queries: dict[str, str] = {}
    answers: dict[str, str] = {}
    question_types: dict[str, str] = {}
    evidence: dict[str, tuple[Evidence, ...]] = {}
    for index, item in enumerate(raw_queries):
        _require_fields(
            item,
            {"query", "answer", "question_type", "evidence_list"},
            "query record",
        )
        query_id = f"mhq-{index:04d}"
        question_type = str(item["question_type"])
        if question_type not in ANSWERABLE_TYPES | {NULL_TYPE}:
            raise ValueError(f"Unknown question type {question_type!r} at {query_id}.")
        annotations: list[Evidence] = []
        for annotation in item["evidence_list"]:
            _require_fields(annotation, {"url", "fact"}, f"evidence for {query_id}")
            url = str(annotation["url"])
            if url not in url_to_doc_id:
                raise ValueError(f"Evidence URL is absent from corpus at {query_id}: {url}")
            annotations.append(
                Evidence(doc_id=url_to_doc_id[url], fact=str(annotation["fact"]))
            )
        if question_type == NULL_TYPE and annotations:
            raise ValueError(f"Null query {query_id} unexpectedly has evidence.")
        if question_type != NULL_TYPE and not annotations:
            raise ValueError(f"Answerable query {query_id} has no evidence.")
        queries[query_id] = str(item["query"])
        answers[query_id] = str(item["answer"])
        question_types[query_id] = question_type
        evidence[query_id] = tuple(annotations)

    dataset = MultiHopDataset(
        corpus=corpus,
        queries=queries,
        answers=answers,
        question_types=question_types,
        evidence=evidence,
        source_hashes={
            "queries_sha256": file_sha256(query_path),
            "corpus_sha256": file_sha256(corpus_path),
        },
    )
    validate_evidence_text(dataset)
    return dataset


def chunk_corpus(
    corpus: dict[str, dict[str, Any]],
    chunk_words: int = 256,
    overlap_words: int = 64,
    strategy: str = "fixed",
    semantic_encoder: SentenceEncoder | None = None,
    semantic_break_percentile: float = 15.0,
    semantic_min_chunk_words: int = 64,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive.")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words).")
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    if strategy != "fixed" and overlap_words:
        raise ValueError("Overlap is supported only by the fixed chunking strategy.")
    if strategy == "semantic" and semantic_encoder is None:
        raise ValueError("Semantic chunking requires a sentence encoder.")
    if not 0.0 <= semantic_break_percentile <= 100.0:
        raise ValueError("semantic_break_percentile must be between 0 and 100.")
    if strategy == "semantic" and (
        semantic_min_chunk_words <= 0 or semantic_min_chunk_words > chunk_words
    ):
        raise ValueError("semantic_min_chunk_words must be in [1, chunk_words].")

    chunks: dict[str, dict[str, Any]] = {}
    chunk_to_doc: dict[str, str] = {}
    for doc_id, document in corpus.items():
        text = str(document["text"])
        if strategy == "fixed":
            bodies = fixed_word_chunks(text, chunk_words, overlap_words)
        elif strategy == "structural":
            bodies = structural_chunks(text, chunk_words)
        else:
            bodies = semantic_chunks(
                text,
                chunk_words=chunk_words,
                min_chunk_words=semantic_min_chunk_words,
                break_percentile=semantic_break_percentile,
                encoder=semantic_encoder,
            )
        metadata_title = format_retrieval_metadata(document)
        word_start = 0
        for chunk_index, body in enumerate(bodies):
            body_words = body.split()
            chunk_id = f"{doc_id}::c{chunk_index:03d}"
            chunks[chunk_id] = {
                **document,
                "title": metadata_title,
                "text": body,
                "article_title": document["title"],
                "parent_doc_id": doc_id,
                "chunk_index": chunk_index,
                "word_start": word_start,
                "word_end": word_start + len(body_words),
                "chunking_strategy": strategy,
            }
            chunk_to_doc[chunk_id] = doc_id
            word_start += len(body_words) - (overlap_words if strategy == "fixed" else 0)
    return chunks, chunk_to_doc


def fixed_word_chunks(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    step = chunk_words - overlap_words
    bodies: list[str] = []
    for start in range(0, len(words), step):
        body = " ".join(words[start : start + chunk_words])
        if body:
            bodies.append(body)
        if start + chunk_words >= len(words):
            break
    return bodies


def structural_chunks(text: str, chunk_words: int) -> list[str]:
    paragraphs = [normalize_space(part) for part in re.split(r"\n\s*\n+", text)]
    units: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        units.extend(fixed_word_chunks(paragraph, chunk_words, 0))
    return pack_units(units, chunk_words)


def semantic_chunks(
    text: str,
    chunk_words: int,
    min_chunk_words: int,
    break_percentile: float,
    encoder: SentenceEncoder,
) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return fixed_word_chunks(text, chunk_words, 0)
    embeddings = np.asarray(
        encoder.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sentences):
        raise ValueError("Semantic encoder returned an unexpected embedding shape.")
    similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    threshold = float(np.percentile(similarities, break_percentile))

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for index, sentence in enumerate(sentences):
        sentence_words = len(sentence.split())
        if sentence_words > chunk_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            chunks.extend(fixed_word_chunks(sentence, chunk_words, 0))
            continue
        exceeds_limit = bool(current and current_words + sentence_words > chunk_words)
        semantic_break = bool(
            current
            and current_words >= min_chunk_words
            and index > 0
            and similarities[index - 1] <= threshold
        )
        if exceeds_limit or semantic_break:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_sentences(text: str) -> list[str]:
    normalized = normalize_space(text)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", normalized)
        if sentence.strip()
    ]


def pack_units(units: list[str], chunk_words: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        unit_words = len(unit.split())
        if current and current_words + unit_words > chunk_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(unit)
        current_words += unit_words
    if current:
        chunks.append(" ".join(current))
    return chunks


def evaluate_evidence_per_query(
    evidence: dict[str, tuple[Evidence, ...]],
    runs: dict[str, dict[str, float]],
    chunk_corpus: dict[str, dict[str, Any]],
    chunk_to_doc: dict[str, str],
    cutoffs: Sequence[int],
) -> dict[str, dict[str, float]]:
    per_query: dict[str, dict[str, float]] = {}
    normalized_chunks = {
        chunk_id: normalize_space(str(chunk["text"]))
        for chunk_id, chunk in chunk_corpus.items()
    }
    for query_id, annotations in evidence.items():
        if not annotations:
            continue
        ranked = [
            chunk_id
            for chunk_id, _ in sorted(
                runs.get(query_id, {}).items(), key=lambda item: (-item[1], item[0])
            )
        ]
        gold_doc_ids = {annotation.doc_id for annotation in annotations}
        query_metrics: dict[str, float] = {}
        for cutoff in cutoffs:
            selected = ranked[:cutoff]
            selected_doc_ids = {chunk_to_doc[chunk_id] for chunk_id in selected}
            covered = [
                annotation
                for annotation in annotations
                if any(
                    chunk_to_doc[chunk_id] == annotation.doc_id
                    and normalize_space(annotation.fact) in normalized_chunks[chunk_id]
                    for chunk_id in selected
                )
            ]
            query_metrics[f"evidence_recall@{cutoff}"] = len(covered) / len(annotations)
            query_metrics[f"evidence_doc_recall@{cutoff}"] = (
                len(gold_doc_ids & selected_doc_ids) / len(gold_doc_ids)
            )
            query_metrics[f"evidence_hit@{cutoff}"] = float(bool(covered))
            query_metrics[f"context_sufficiency@{cutoff}"] = float(
                len(covered) == len(annotations)
            )
        per_query[query_id] = query_metrics
    return per_query


def mean_metrics(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    names = sorted({name for metrics in per_query.values() for name in metrics})
    return {
        name: mean(metrics[name] for metrics in per_query.values())
        for name in names
    }


def dataset_audit(dataset: MultiHopDataset) -> dict[str, Any]:
    evidence_counts = [len(items) for items in dataset.evidence.values()]
    unique_doc_counts = [
        len({item.doc_id for item in items}) for items in dataset.evidence.values()
    ]
    answerable_ids = dataset.answerable_query_ids
    article_word_counts = [
        len(str(document["text"]).split()) for document in dataset.corpus.values()
    ]
    type_counts = Counter(dataset.question_types.values())
    answerable_evidence_distribution = Counter(
        len(dataset.evidence[query_id]) for query_id in answerable_ids
    )
    answerable_document_distribution = Counter(
        len({item.doc_id for item in dataset.evidence[query_id]})
        for query_id in answerable_ids
    )
    source_name_matches = []
    for query_id in answerable_ids:
        query = dataset.queries[query_id].lower()
        sources = {
            str(dataset.corpus[item.doc_id].get("source", "")).lower()
            for item in dataset.evidence[query_id]
        }
        matches = [bool(source and source in query) for source in sources]
        source_name_matches.append(matches)
    return {
        "question_count": len(dataset.queries),
        "question_type_counts": dict(sorted(type_counts.items())),
        "answerable_query_count": len(answerable_ids),
        "null_query_count": len(dataset.null_query_ids),
        "corpus_document_count": len(dataset.corpus),
        "corpus_granularity": "full_news_articles",
        "article_words_mean": mean(article_word_counts),
        "article_words_median": median(article_word_counts),
        "article_words_min": min(article_word_counts),
        "article_words_max": max(article_word_counts),
        "evidence_annotation_count": sum(evidence_counts),
        "evidence_annotations_per_answerable_distribution": dict(
            sorted(answerable_evidence_distribution.items())
        ),
        "mean_evidence_annotations_all_queries": mean(evidence_counts),
        "mean_evidence_annotations_answerable": mean(
            len(dataset.evidence[query_id]) for query_id in answerable_ids
        ),
        "mean_unique_evidence_documents_all_queries": mean(unique_doc_counts),
        "mean_unique_evidence_documents_answerable": mean(
            len({item.doc_id for item in dataset.evidence[query_id]})
            for query_id in answerable_ids
        ),
        "unique_evidence_documents_per_answerable_distribution": dict(
            sorted(answerable_document_distribution.items())
        ),
        "queries_with_repeated_evidence_document": sum(
            len({item.doc_id for item in items}) != len(items)
            for items in dataset.evidence.values()
        ),
        "answerable_queries_naming_any_evidence_source": sum(
            any(matches) for matches in source_name_matches
        ),
        "answerable_queries_naming_all_evidence_sources": sum(
            all(matches) for matches in source_name_matches
        ),
        "all_evidence_urls_resolved": True,
        "all_evidence_facts_found_in_source_articles": True,
        **dataset.source_hashes,
    }


def validate_evidence_text(dataset: MultiHopDataset) -> None:
    for query_id, annotations in dataset.evidence.items():
        for annotation in annotations:
            body = normalize_space(str(dataset.corpus[annotation.doc_id]["text"]))
            if normalize_space(annotation.fact) not in body:
                raise ValueError(
                    f"Evidence fact for {query_id} is absent from {annotation.doc_id}."
                )


def document_id(url: str) -> str:
    return f"mhdoc-{hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]}"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_retrieval_metadata(document: dict[str, Any]) -> str:
    fields = (
        ("title", document.get("title")),
        ("source", document.get("source")),
        ("published_at", document.get("published_at")),
    )
    return "\n".join(f"{name}: {value}" for name, value in fields if value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    temporary_path = path.with_suffix(path.suffix + ".download")
    urllib.request.urlretrieve(url, temporary_path)
    temporary_path.replace(path)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a JSON list of objects in {path}.")
    return value


def _require_fields(item: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - item.keys())
    if missing:
        raise ValueError(f"Missing fields in {label}: {missing}")
