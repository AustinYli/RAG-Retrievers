from __future__ import annotations

import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm.auto import tqdm

from rag_bench.load import corpus_text


TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
DEFAULT_ANALYZER = "word-lower-v1"
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_BM25_EPSILON = 0.25


def tokenize(text: str, analyzer: str = DEFAULT_ANALYZER) -> list[str]:
    words = TOKEN_RE.findall(text.lower())
    if analyzer == "word-lower-v1":
        return words
    if analyzer == "word-lower-stop-v1":
        return [word for word in words if word not in ENGLISH_STOP_WORDS]
    if analyzer == "word-lower-porter-v1":
        return [_porter_stem(word) for word in words]
    if analyzer == "word-lower-stop-porter-v1":
        return [_porter_stem(word) for word in words if word not in ENGLISH_STOP_WORDS]
    if analyzer == "word-lower-bigram-v1":
        bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:], strict=False)]
        return [*words, *bigrams]
    raise ValueError(f"Unknown analyzer: {analyzer}")


@dataclass(frozen=True)
class CorpusView:
    doc_ids: list[str]
    texts: list[str]

    @classmethod
    def from_corpus(cls, corpus: dict[str, dict[str, Any]]) -> "CorpusView":
        doc_ids = list(corpus.keys())
        texts = [corpus_text(corpus[doc_id]) for doc_id in doc_ids]
        return cls(doc_ids=doc_ids, texts=texts)


class BM25Retriever:
    def __init__(
        self,
        corpus: dict[str, dict[str, Any]],
        cache_dir: str | Path,
        dataset: str,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
        epsilon: float = DEFAULT_BM25_EPSILON,
        analyzer: str = DEFAULT_ANALYZER,
    ):
        self.view = CorpusView.from_corpus(corpus)
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.analyzer = analyzer
        fingerprint = corpus_fingerprint(self.view.doc_ids, self.view.texts)
        params = f"analyzer={analyzer}_k1={k1}_b={b}_eps={epsilon}"
        self.cache_path = Path(cache_dir) / "bm25" / f"{dataset}_{len(self.view.doc_ids)}_{fingerprint}_{_slug(params)}.pkl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = self._load_or_build()

    def _load_or_build(self) -> BM25Okapi:
        if self.cache_path.exists():
            with self.cache_path.open("rb") as file:
                return pickle.load(file)

        tokenized = [tokenize(text, self.analyzer) for text in tqdm(self.view.texts, desc="Tokenizing corpus")]
        index = BM25Okapi(tokenized, k1=self.k1, b=self.b, epsilon=self.epsilon)
        with self.cache_path.open("wb") as file:
            pickle.dump(index, file)
        return index

    def search(self, queries: dict[str, str], top_k: int) -> dict[str, dict[str, float]]:
        runs, _ = self.search_with_latencies(queries, top_k)
        return runs

    def search_with_latencies(
        self,
        queries: dict[str, str],
        top_k: int,
    ) -> tuple[dict[str, dict[str, float]], list[float]]:
        import time

        runs: dict[str, dict[str, float]] = {}
        latencies: list[float] = []
        safe_top_k = min(top_k, len(self.view.doc_ids))
        for query_id, query in tqdm(queries.items(), desc="BM25 search"):
            started = time.perf_counter()
            scores = self.index.get_scores(tokenize(query, self.analyzer))
            if safe_top_k == len(scores):
                top_indices = np.argsort(-scores)
            else:
                top_indices = np.argpartition(-scores, safe_top_k - 1)[:safe_top_k]
                top_indices = top_indices[np.argsort(-scores[top_indices])]
            runs[query_id] = {
                self.view.doc_ids[int(index)]: float(scores[int(index)])
                for index in top_indices
                if scores[int(index)] > 0
            }
            latencies.append(time.perf_counter() - started)
        return runs, latencies


class DenseRetriever:
    def __init__(
        self,
        corpus: dict[str, dict[str, Any]],
        cache_dir: str | Path,
        dataset: str,
        model_name: str,
        batch_size: int = 64,
        device: str | None = None,
        revision: str | None = None,
        max_seq_length: int | None = None,
        normalize_embeddings: bool = True,
    ):
        self.view = CorpusView.from_corpus(corpus)
        self.cache_dir = Path(cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.model_name = model_name
        self.model_revision = revision or ""
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device, revision=revision)
        if max_seq_length is not None:
            self.model.max_seq_length = int(max_seq_length)
        self.max_seq_length = int(getattr(self.model, "max_seq_length", 0) or 0)
        self.device = str(getattr(self.model, "device", device or ""))
        self.doc_embeddings = self._load_or_encode_docs()
        self.index: faiss.IndexFlatIP | None = None

    def search(
        self,
        queries: dict[str, str],
        top_k: int,
        index_backend: str = "numpy",
    ) -> dict[str, dict[str, float]]:
        query_ids = list(queries.keys())
        query_texts = [prefix_query(self.model_name, queries[query_id]) for query_id in query_ids]
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")

        safe_top_k = min(top_k, len(self.view.doc_ids))
        if index_backend == "faiss":
            if self.index is None:
                self.index = self._build_index(self.doc_embeddings)
            scores, indices = self.index.search(query_embeddings, safe_top_k)
        elif index_backend == "numpy":
            scores, indices = exact_numpy_search(query_embeddings, self.doc_embeddings, safe_top_k)
        else:
            raise ValueError(f"Unknown dense index backend: {index_backend}")

        runs: dict[str, dict[str, float]] = {}
        for row_index, query_id in enumerate(query_ids):
            runs[query_id] = {
                self.view.doc_ids[int(doc_index)]: float(score)
                for score, doc_index in zip(scores[row_index], indices[row_index], strict=False)
                if doc_index >= 0
            }
        return runs

    def search_with_latencies(
        self,
        queries: dict[str, str],
        top_k: int,
        index_backend: str = "numpy",
    ) -> tuple[dict[str, dict[str, float]], list[float]]:
        import time

        safe_top_k = min(top_k, len(self.view.doc_ids))
        runs: dict[str, dict[str, float]] = {}
        latencies: list[float] = []
        for query_id, query in tqdm(queries.items(), desc="Dense search"):
            started = time.perf_counter()
            query_embedding = self.model.encode(
                [prefix_query(self.model_name, query)],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            ).astype("float32")
            if index_backend == "faiss":
                if self.index is None:
                    self.index = self._build_index(self.doc_embeddings)
                scores, indices = self.index.search(query_embedding, safe_top_k)
            elif index_backend == "numpy":
                scores, indices = exact_numpy_search(query_embedding, self.doc_embeddings, safe_top_k)
            else:
                raise ValueError(f"Unknown dense index backend: {index_backend}")

            runs[query_id] = {
                self.view.doc_ids[int(doc_index)]: float(score)
                for score, doc_index in zip(scores[0], indices[0], strict=False)
                if doc_index >= 0
            }
            latencies.append(time.perf_counter() - started)
        return runs, latencies

    def _load_or_encode_docs(self) -> np.ndarray:
        fingerprint = corpus_fingerprint(self.view.doc_ids, self.view.texts)
        params = f"seq={self.max_seq_length}_normalize={self.normalize_embeddings}"
        if self.model_revision:
            params += f"_revision={self.model_revision}"
        cache_path = (
            self.cache_dir
            / f"{self.dataset}_{_model_slug(self.model_name)}_{len(self.view.doc_ids)}_{fingerprint}_{_slug(params)}.npy"
        )
        if cache_path.exists():
            return np.load(cache_path).astype("float32")

        texts = [prefix_document(self.model_name, text) for text in self.view.texts]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")
        np.save(cache_path, embeddings)
        return embeddings

    @staticmethod
    def _build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index


def exact_numpy_search(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = query_embeddings @ doc_embeddings.T
    if top_k == scores.shape[1]:
        indices = np.argsort(-scores, axis=1)
    else:
        indices = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        row_indices = np.arange(scores.shape[0])[:, None]
        order = np.argsort(-scores[row_indices, indices], axis=1)
        indices = indices[row_indices, order]

    row_indices = np.arange(scores.shape[0])[:, None]
    sorted_scores = scores[row_indices, indices]
    return sorted_scores.astype("float32"), indices.astype("int64")


def hybrid_rrf(
    bm25_runs: dict[str, dict[str, float]],
    dense_runs: dict[str, dict[str, float]],
    top_k: int,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> dict[str, dict[str, float]]:
    query_ids = sorted(set(bm25_runs) | set(dense_runs))
    combined, _ = hybrid_rrf_with_latencies(
        bm25_runs,
        dense_runs,
        query_ids,
        top_k,
        rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )
    return combined


def hybrid_rrf_with_latencies(
    bm25_runs: dict[str, dict[str, float]],
    dense_runs: dict[str, dict[str, float]],
    query_ids: list[str],
    top_k: int,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> tuple[dict[str, dict[str, float]], list[float]]:
    import time

    combined: dict[str, dict[str, float]] = {}
    latencies: list[float] = []
    for query_id in query_ids:
        started = time.perf_counter()
        scores: dict[str, float] = {}
        for run, weight in (
            (bm25_runs.get(query_id, {}), bm25_weight),
            (dense_runs.get(query_id, {}), dense_weight),
        ):
            ranked = sorted(run.items(), key=lambda item: (-item[1], item[0]))
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
        combined[query_id] = dict(
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        )
        latencies.append(time.perf_counter() - started)
    return combined, latencies


def prefix_query(model_name: str, text: str) -> str:
    lowered = model_name.lower()
    if "e5-" in lowered or "/e5-" in lowered:
        return f"query: {text}"
    if "bge-" in lowered:
        return f"Represent this sentence for searching relevant passages: {text}"
    return text


def prefix_document(model_name: str, text: str) -> str:
    lowered = model_name.lower()
    if "e5-" in lowered or "/e5-" in lowered:
        return f"passage: {text}"
    return text


def _model_slug(model_name: str) -> str:
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()
    return f"{slug}-{digest}"


def _slug(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-zA-Z0-9.=-]+", "-", text).strip("-").lower()
    return f"{slug}-{digest}"


def corpus_fingerprint(doc_ids: list[str], texts: list[str]) -> str:
    digest = hashlib.sha1()
    for doc_id, text in zip(doc_ids, texts, strict=True):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _porter_stem(word: str) -> str:
    if len(word) <= 2:
        return word

    def cons(index: int) -> bool:
        char = word[index]
        if char in "aeiou":
            return False
        if char == "y":
            return index == 0 or not cons(index - 1)
        return True

    def measure(stem: str) -> int:
        count = 0
        in_vowel_run = False
        for index, _ in enumerate(stem):
            is_cons = _is_consonant(stem, index)
            if is_cons:
                if in_vowel_run:
                    count += 1
                in_vowel_run = False
            else:
                in_vowel_run = True
        return count

    def has_vowel(stem: str) -> bool:
        return any(not _is_consonant(stem, index) for index in range(len(stem)))

    def double_consonant(stem: str) -> bool:
        return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)

    def cvc(stem: str) -> bool:
        if len(stem) < 3:
            return False
        return (
            _is_consonant(stem, len(stem) - 1)
            and not _is_consonant(stem, len(stem) - 2)
            and _is_consonant(stem, len(stem) - 3)
            and stem[-1] not in "wxy"
        )

    def replace_suffix(suffix: str, replacement: str, min_measure: int = 0) -> bool:
        nonlocal word
        if not word.endswith(suffix):
            return False
        stem = word[: -len(suffix)]
        if measure(stem) > min_measure:
            word = stem + replacement
            return True
        return False

    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    flag = False
    for suffix in ("eed", "ed", "ing"):
        if not word.endswith(suffix):
            continue
        stem = word[: -len(suffix)]
        if suffix == "eed":
            if measure(stem) > 0:
                word = word[:-1]
            break
        if has_vowel(stem):
            word = stem
            flag = True
            break
    if flag:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif measure(word) == 1 and cvc(word):
            word += "e"

    if word.endswith("y") and has_vowel(word[:-1]):
        word = word[:-1] + "i"

    for suffix, replacement in (
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("logi", "log"),
    ):
        if replace_suffix(suffix, replacement):
            break

    for suffix, replacement in (
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ):
        if replace_suffix(suffix, replacement):
            break

    for suffix in (
        "al",
        "ance",
        "ence",
        "er",
        "ic",
        "able",
        "ible",
        "ant",
        "ement",
        "ment",
        "ent",
        "ou",
        "ism",
        "ate",
        "iti",
        "ous",
        "ive",
        "ize",
    ):
        if replace_suffix(suffix, "", min_measure=1):
            break
    else:
        for suffix in ("ion",):
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if stem.endswith(("s", "t")) and measure(stem) > 1:
                    word = stem
                break

    if word.endswith("e"):
        stem = word[:-1]
        stem_measure = measure(stem)
        if stem_measure > 1 or (stem_measure == 1 and not cvc(stem)):
            word = stem
    if measure(word) > 1 and double_consonant(word) and word.endswith("l"):
        word = word[:-1]

    return word


def _is_consonant(word: str, index: int) -> bool:
    char = word[index]
    if char in "aeiou":
        return False
    if char == "y":
        return index == 0 or not _is_consonant(word, index - 1)
    return True
