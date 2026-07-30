"""Independent one-round peer review for each grader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from consensus.providers import ProviderCallResult, invoke_model
from consensus.schemas import (
    ValidatedCrossReview,
    clean_json_object,
    validate_cross_review,
)
from grading.spring_2025_prompt import SPRING_2025_RUBRIC


REVIEW_MAX_TOKENS = 6000
REVIEW_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class CrossReviewResult:
    validated: ValidatedCrossReview
    raw_text: str
    raw_response: Any
    request_metadata: dict[str, Any]


def _peer_projection(grading: Mapping[str, Any]) -> dict[str, Any]:
    from consensus.comparison import COMPARISON_FIELDS, get_path

    return {
        "model": grading.get("model"),
        "compared_values": {
            field.path: get_path(grading, field.path) for field in COMPARISON_FIELDS
        },
        "criterion_explanations": {
            field.path.removesuffix(".score"): get_path(
                grading,
                field.path.removesuffix(".score") + ".explanation",
            )
            for field in COMPARISON_FIELDS
            if field.disagreement_type == "criterion_score"
        },
    }


def build_cross_review_prompt(
    *,
    reviewer_model: str,
    reviewed_peer_model: str,
    initial_own: Mapping[str, Any],
    initial_peer: Mapping[str, Any],
    disagreements: list[dict[str, Any]],
) -> str:
    return (
        "Perform one independent cross-review of a medical concept-map grading. The "
        "original image is authoritative; the peer output is critique, not an answer key. "
        "Review only disputed fields, except that a logically related domain or overall "
        "decision may be reconsidered. Never average, compromise merely to agree, prefer "
        "the longer answer, or defer automatically. Do not change undisputed criterion "
        "scores. Revise only when visible image evidence and the exact rubric support it; "
        "otherwise defend the original value. Missing detail is not missing reasoning; "
        "terminology is not integration; do not infer invisible relationships. Score 4 "
        "requires the full criterion intent, 3 substantial evidence with a meaningful "
        "limitation, and 2 a genuinely underdeveloped central relationship. Preserve the "
        "strong-map versus weak-map calibration.\n\n"
        + SPRING_2025_RUBRIC
        + "\n\nREVIEWER\n"
        + reviewer_model
        + "\nPEER\n"
        + reviewed_peer_model
        + "\n\nIMMUTABLE OWN INITIAL GRADING\n"
        + json.dumps(initial_own, separators=(",", ":"), ensure_ascii=False)
        + "\n\nPEER INITIAL GRADING SUMMARY\n"
        + json.dumps(_peer_projection(initial_peer), separators=(",", ":"), ensure_ascii=False)
        + "\n\nINITIAL DISAGREEMENTS\n"
        + json.dumps(disagreements, separators=(",", ":"), ensure_ascii=False)
        + "\n\nReturn one raw JSON object with reviewer_model, reviewed_peer_model, "
        "reviewed_grading (the complete current grading schema), field_reviews, "
        "changed_field_paths, defended_field_paths, and review_confidence. Each field review "
        "must include path, initial_own_value, peer_value, reviewed_own_value, action "
        "(revised, defended, agreed_with_peer, or unresolved_uncertainty), and a specific "
        "reason. Every changed field must be declared. Preserve all grounded multimodal "
        "fields unless the image requires an evidence correction. Silently verify that "
        "undisputed scores did not change. No Markdown or prose outside JSON."
    )


def run_cross_review(
    *,
    reviewer_model: str,
    reviewed_peer_model: str,
    provider: str,
    model_id: str,
    image_base64: str,
    initial_own: Mapping[str, Any],
    initial_peer: Mapping[str, Any],
    initial_comparison: Mapping[str, Any],
    invoke: Callable[..., ProviderCallResult] = invoke_model,
) -> CrossReviewResult:
    disagreements = list(initial_comparison.get("disagreements", []))
    prompt = build_cross_review_prompt(
        reviewer_model=reviewer_model,
        reviewed_peer_model=reviewed_peer_model,
        initial_own=initial_own,
        initial_peer=initial_peer,
        disagreements=disagreements,
    )
    call = invoke(
        provider=provider,
        model_id=model_id,
        prompt=prompt,
        image_base64=image_base64,
        max_tokens=REVIEW_MAX_TOKENS,
        timeout_seconds=REVIEW_TIMEOUT_SECONDS,
    )
    payload = clean_json_object(call.raw_text)
    validated = validate_cross_review(
        payload,
        reviewer_model=reviewer_model,
        reviewed_peer_model=reviewed_peer_model,
        initial_own=initial_own,
        initial_peer=initial_peer,
        initial_disagreement_paths={
            str(item["path"]) for item in disagreements
        },
    )
    metadata = {
        **call.metadata,
        "image_resent": True,
        "review_call_count": 1,
    }
    return CrossReviewResult(validated, call.raw_text, call.raw_response, metadata)
