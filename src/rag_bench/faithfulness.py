from __future__ import annotations

import re
from typing import Any

import numpy as np


def split_claims(answer: str) -> list[str]:
    return [
        claim.strip()
        for claim in re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
        if claim.strip()
    ]


def entailment_label_index(model: Any) -> int:
    id2label = getattr(getattr(model, "model", None), "config", None)
    labels = getattr(id2label, "id2label", {})
    for index, label in labels.items():
        if str(label).lower() == "entailment":
            return int(index)
    raise ValueError("NLI model config does not identify an entailment label.")


def claim_support_scores(
    context: str,
    answer: str,
    model: Any,
    batch_size: int = 32,
    split_sentences: bool = True,
) -> list[float]:
    claims = split_claims(answer) if split_sentences else [answer.strip()]
    claims = [claim for claim in claims if claim]
    if not claims:
        return []
    context_blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
    if not context_blocks:
        return [0.0] * len(claims)
    logits = np.asarray(
        model.predict(
            [
                (context_block, claim)
                for claim in claims
                for context_block in context_blocks
            ],
            batch_size=batch_size,
            apply_softmax=True,
            show_progress_bar=False,
        )
    )
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    entailment_index = entailment_label_index(model)
    entailment_scores = logits[:, entailment_index].reshape(
        len(claims), len(context_blocks)
    )
    return [float(row.max()) for row in entailment_scores]


def unsupported_claim_rate(scores: list[float], threshold: float = 0.5) -> float:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one.")
    if not scores:
        return 0.0
    return sum(score < threshold for score in scores) / len(scores)


def cohen_kappa_binary(expected: list[bool], observed: list[bool]) -> float:
    if len(expected) != len(observed):
        raise ValueError("Kappa inputs must have equal lengths.")
    if not expected:
        return 0.0
    agreement = sum(left == right for left, right in zip(expected, observed, strict=True))
    observed_agreement = agreement / len(expected)
    expected_true = sum(expected) / len(expected)
    observed_true = sum(observed) / len(observed)
    chance_agreement = (
        expected_true * observed_true
        + (1.0 - expected_true) * (1.0 - observed_true)
    )
    if chance_agreement == 1.0:
        return 1.0 if observed_agreement == 1.0 else 0.0
    return (observed_agreement - chance_agreement) / (1.0 - chance_agreement)
