"""Pure helpers used by the active Streamlit consensus orchestration."""

from __future__ import annotations

import base64
import copy
from pathlib import Path
from typing import Iterable

from consensus.comparison import compare_gradings
from interface.grading_runner import EvaluationOutcome, EvaluationResult, GradingError


INITIAL_MODEL_KEYS = {
    "Gemma": "gemma",
    "Llama 3.2 90B Vision": "llama",
}


def successful_results(
    results: Iterable[EvaluationOutcome],
) -> dict[str, EvaluationResult]:
    return {
        result.model_name: result
        for result in results
        if isinstance(result, EvaluationResult)
    }


def immutable_initial_results(
    results: Iterable[EvaluationOutcome],
) -> dict[str, dict]:
    """Copy successful initial grades so review stages cannot alter the baseline."""
    successes = successful_results(results)
    return {
        key: copy.deepcopy(successes[label].data)
        for label, key in INITIAL_MODEL_KEYS.items()
        if label in successes
    }


def consensus_ready(results: Iterable[EvaluationOutcome]) -> bool:
    """Consensus requires both successful, validated independent outcomes."""
    successes = successful_results(results)
    return all(label in successes for label in INITIAL_MODEL_KEYS)


def exact_request_image_inputs(
    results: Iterable[EvaluationOutcome],
) -> dict[str, str]:
    """Encode the exact persisted JPEG used by each independent grader."""
    successes = successful_results(results)
    encoded: dict[str, str] = {}
    for label, key in INITIAL_MODEL_KEYS.items():
        result = successes.get(label)
        if result is None:
            continue
        image_path = (
            Path(result.source_image_path) if result.source_image_path is not None else None
        )
        if image_path is None or not image_path.exists():
            raise GradingError(
                f"The original {result.model_name} request image is unavailable for consensus."
            )
        encoded[key] = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return encoded


def fallback_comparison_export(
    map_file: str,
    initial: dict[str, dict],
) -> dict:
    gemma = initial.get("gemma")
    llama = initial.get("llama")
    comparison = (
        compare_gradings(gemma, llama)
        if isinstance(gemma, dict) and isinstance(llama, dict)
        else None
    )
    return {
        "map_file": map_file,
        "initial_results": copy.deepcopy(initial),
        "initial_comparison": comparison,
        "cross_reviews": {"gemma": None, "llama": None},
        "post_review_comparison": None,
        "consensus": {
            "consensus_status": "unavailable",
            "consensus_grading": None,
            "criterion_resolutions": [],
            "unresolved_disagreements": [],
        },
    }
