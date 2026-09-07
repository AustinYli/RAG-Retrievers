from __future__ import annotations

import hashlib
import json
import re
import string
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any


ANSWER_NORMALIZER_VERSION = "squad-lower-ascii-punct-articles-whitespace-v1"
RESPONSE_PARSER_VERSION = "answer-claim-two-line-complete-max25-v2"
MAX_CLAIM_WORDS = 25
PROMPT_DEVELOPMENT_QUERY_COUNT = 20
EVALUATION_PARTITION_VERSION = "exclude-first-20-answerable-prompt-development-v1"
EVALUATION_SAMPLER_VERSION = "sha256-seed-query-id-v1"
ABSTENTION_INSTRUCTION_ON = (
    "If the evidence does not support an answer, use 'Insufficient information.' "
    "as the Answer and state in the Claim that the evidence is insufficient."
)
DEFAULT_ABSTENTION_PATTERNS = (
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\binsufficient (?:evidence|information|context)\b",
    r"\bnot enough (?:evidence|information|context)\b",
    r"\bcannot (?:answer|determine)\b",
    r"\bcan't (?:answer|determine)\b",
    r"\b(?:evidence|context) (?:does not|doesn't) (?:provide|contain|support)\b",
    r"\bno (?:supporting|relevant) (?:evidence|information|context)\b",
    r"\bnot (?:provided|stated|specified|mentioned) in (?:the )?(?:evidence|context)\b",
)


@dataclass(frozen=True)
class OllamaConfig:
    model: str
    temperature: float = 0.0
    seed: int = 13
    context_window: int = 8192
    max_new_tokens: int = 64
    base_url: str = "http://127.0.0.1:11434"


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_answer_and_claim(response: str) -> tuple[str, str, bool]:
    answer_match = re.search(r"^Answer:\s*(.+?)\s*$", response, flags=re.MULTILINE)
    claim_match = re.search(r"^Claim:\s*(.+?)\s*$", response, flags=re.MULTILINE)
    if answer_match and claim_match:
        answer = answer_match.group(1).strip()
        claim = claim_match.group(1).strip()
        valid_structure = bool(
            re.fullmatch(
                r"Answer:\s*[^\n]+\nClaim:\s*[^\n]+",
                response.strip(),
            )
        )
        valid_claim = (
            len(claim.split()) <= MAX_CLAIM_WORDS
            and bool(re.search(r"[.!?]['\")\]]*$", claim))
        )
        return answer, claim, valid_structure and valid_claim
    fallback = response.strip()
    return fallback, fallback, False


def is_abstention(
    answer: str, patterns: tuple[str, ...] = DEFAULT_ABSTENTION_PATTERNS
) -> bool:
    return any(re.search(pattern, answer, flags=re.IGNORECASE) for pattern in patterns)


def abstention_metrics(
    predictions: dict[str, str], question_types: dict[str, str]
) -> dict[str, float | None]:
    null_ids = [query_id for query_id in predictions if question_types[query_id] == "null_query"]
    answerable_ids = [
        query_id for query_id in predictions if question_types[query_id] != "null_query"
    ]
    null_rate = _rate(is_abstention(predictions[query_id]) for query_id in null_ids)
    false_rate = _rate(
        is_abstention(predictions[query_id]) for query_id in answerable_ids
    )
    return {
        "null_abstention_rate": null_rate,
        "answerable_false_abstention_rate": false_rate,
        "abstention_separation": (
            null_rate - false_rate
            if null_rate is not None and false_rate is not None
            else None
        ),
    }


def prompt_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def stable_query_sample(query_ids: list[str], size: int, seed: int) -> list[str]:
    if size <= 0 or size > len(query_ids):
        raise ValueError("Sample size must be positive and no larger than the query pool.")

    def rank(query_id: str) -> str:
        return hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).hexdigest()

    selected = set(sorted(query_ids, key=rank)[:size])
    return [query_id for query_id in query_ids if query_id in selected]


def ollama_generate(
    prompt: str,
    config: OllamaConfig,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/api/generate",
        data=json.dumps(
            {
                "model": config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "num_ctx": config.context_window,
                    "num_predict": config.max_new_tokens,
                },
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
    if not isinstance(result, dict) or "response" not in result:
        raise ValueError("Ollama returned an unexpected response payload.")
    return result


def ollama_show(config: OllamaConfig, timeout_seconds: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/api/show",
        data=json.dumps({"model": config.model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Ollama returned an unexpected model metadata payload.")
    return result


def ollama_model_metadata(
    config: OllamaConfig, timeout_seconds: float = 30.0
) -> dict[str, Any]:
    result = ollama_show(config, timeout_seconds=timeout_seconds)
    with urllib.request.urlopen(
        f"{config.base_url.rstrip('/')}/api/tags", timeout=timeout_seconds
    ) as response:
        tags = json.load(response)
    requested = config.model.removesuffix(":latest")
    match = next(
        (
            item
            for item in tags.get("models", [])
            if str(item.get("name", "")).removesuffix(":latest") == requested
            or str(item.get("model", "")).removesuffix(":latest") == requested
        ),
        {},
    )
    result["manifest_digest"] = match.get("digest", "")
    with urllib.request.urlopen(
        f"{config.base_url.rstrip('/')}/api/version", timeout=timeout_seconds
    ) as response:
        version = json.load(response)
    result["ollama_version"] = version.get("version", "")
    return result


def _rate(values) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None
