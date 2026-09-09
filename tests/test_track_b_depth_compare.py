import json
from types import SimpleNamespace

import pytest

from rag_bench.multihop import Evidence
from scripts.compare_track_b_depths import (
    build_knee_rows,
    build_transition_rows,
    evaluate_depth_artifact,
    validate_depth_controls,
    values_for_stratum,
)


def run_row(depth: int, budget: int) -> dict[str, str]:
    return {
        "mode": "retrieved",
        "retrieved_top_k": str(depth),
        "max_context_words": str(budget),
        "context_packing": "balanced",
        "retriever_run_name": "multihop_bm25_stem_w256_o64",
        "evaluation_sample_size": "300",
        "evaluation_sample_seed": "13",
        "context_window": "16384",
        "retriever_run_id": "bm25-id",
        "selected_query_ids_sha256": "sample",
    }


def test_depth_comparison_allows_only_depth_and_budget_to_change():
    rows = [
        run_row(3, 768),
        run_row(5, 1280),
        run_row(10, 2560),
        run_row(20, 5120),
    ]

    validate_depth_controls(rows)

    rows[2]["selected_query_ids_sha256"] = "other"
    with pytest.raises(ValueError, match="selected_query_ids_sha256"):
        validate_depth_controls(rows)


def test_depth_comparison_rejects_wrong_budget():
    rows = [
        run_row(3, 770),
        run_row(5, 1280),
        run_row(10, 2560),
        run_row(20, 5120),
    ]

    with pytest.raises(ValueError, match="context budgets"):
        validate_depth_controls(rows)


def test_depth_comparison_rejects_truncating_context_window():
    rows = [
        run_row(3, 768),
        run_row(5, 1280),
        run_row(10, 2560),
        run_row(20, 5120),
    ]
    rows[3]["context_window"] = "8192"

    with pytest.raises(ValueError, match="16,384"):
        validate_depth_controls(rows)


def test_depth_metrics_use_actual_rendered_context_and_stratify(tmp_path):
    artifact = tmp_path / "run.jsonl"
    artifact.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "q1",
                        "context": "alpha fact is here",
                        "exact_match": 1,
                        "token_f1": 1,
                    }
                ),
                json.dumps(
                    {
                        "query_id": "q2",
                        "context": "only beta fact",
                        "exact_match": 0,
                        "token_f1": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "query_id": "q3",
                        "context": "",
                        "exact_match": 0,
                        "token_f1": 0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        question_types={
            "q1": "inference_query",
            "q2": "temporal_query",
            "q3": "null_query",
        },
        evidence={
            "q1": (Evidence(doc_id="d1", fact="alpha fact"),),
            "q2": (
                Evidence(doc_id="d2", fact="beta fact"),
                Evidence(doc_id="d3", fact="missing fact"),
            ),
        },
    )

    values = evaluate_depth_artifact(artifact, dataset)

    assert values["evidence_recall_at_budget"] == {"q1": 1.0, "q2": 0.5}
    assert values["context_sufficiency_at_budget"] == {"q1": 1.0, "q2": 0.0}
    assert values_for_stratum(values["exact_match"], dataset, "inference_query") == {
        "q1": 1.0
    }


def test_knee_rows_are_explicitly_exploratory():
    rows = [
        {"run_name": f"k{k}", "run_key": str(k), "artifact_path": f"k{k}.jsonl"}
        for k in (3, 5, 10, 20)
    ]
    metrics = {
        "exact_match": {"q1": 1.0},
        "token_f1": {"q1": 1.0},
        "evidence_recall_at_budget": {"q1": 1.0},
        "context_sufficiency_at_budget": {"q1": 1.0},
    }
    per_run = [metrics, metrics, metrics, metrics]
    dataset = SimpleNamespace(question_types={"q1": "inference_query"})

    output = build_knee_rows(rows, per_run, dataset, samples=10, seed=13)

    assert len(output) == 4
    assert {row["comparison"] for row in output} == {"k10_to_k20"}
    assert {row["analysis_status"] for row in output} == {
        "exploratory_not_in_preregistered_holm_families"
    }


def test_transition_rows_capture_new_evidence_and_answer_flips():
    rows = [{"run_key": str(k)} for k in (3, 5, 10, 20)]

    def metrics(em: tuple[float, float], sufficient: tuple[float, float]):
        return {
            "exact_match": {"q1": em[0], "q2": em[1]},
            "context_sufficiency_at_budget": {
                "q1": sufficient[0],
                "q2": sufficient[1],
            },
        }

    per_run = [
        metrics((0, 1), (0, 0)),
        metrics((1, 1), (1, 0)),
        metrics((1, 0), (1, 1)),
        metrics((1, 0), (1, 1)),
    ]

    output = build_transition_rows(rows, per_run)

    assert output[0]["correctness_gains"] == 1
    assert output[0]["newly_sufficient_queries"] == 1
    assert output[0]["newly_sufficient_em_before"] == 0
    assert output[0]["newly_sufficient_em_after"] == 1
    assert output[1]["correctness_losses"] == 1
    assert output[1]["lost_sufficient_queries"] == 0
