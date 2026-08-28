import numpy as np

from rag_bench.retrievers import exact_numpy_search, hybrid_rrf, prefix_document, prefix_query, tokenize


def test_model_prefixes():
    assert prefix_query("intfloat/e5-base-v2", "hello") == "query: hello"
    assert prefix_document("intfloat/e5-base-v2", "hello") == "passage: hello"

    assert (
        prefix_query("BAAI/bge-base-en-v1.5", "hello")
        == "Represent this sentence for searching relevant passages: hello"
    )
    assert prefix_document("BAAI/bge-base-en-v1.5", "hello") == "hello"


def test_exact_numpy_search_sorts_descending():
    docs = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
    queries = np.array([[1.0, 0.0]], dtype="float32")

    scores, indices = exact_numpy_search(queries, docs, top_k=2)

    assert indices.tolist() == [[0, 2]]
    assert scores.tolist() == [[1.0, 0.800000011920929]]


def test_hybrid_rrf_combines_rankings():
    bm25 = {"q1": {"d1": 9.0, "d2": 8.0}}
    dense = {"q1": {"d2": 1.0, "d3": 0.9}}

    combined = hybrid_rrf(bm25, dense, top_k=3, rrf_k=60)

    assert list(combined["q1"]) == ["d2", "d1", "d3"]


def test_porter_analyzer_stems_terms():
    assert tokenize("relational running studies", "word-lower-porter-v1") == [
        "relat",
        "run",
        "studi",
    ]
    assert tokenize("the running study", "word-lower-stop-porter-v1") == ["run", "studi"]
