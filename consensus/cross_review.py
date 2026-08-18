"""Independent one-round peer review for each grader."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from consensus.providers import ProviderCallResult, invoke_model
from consensus.schemas import (
    ConsensusValidationError,
    ValidatedCrossReview,
    clean_json_object,
    validate_cross_review,
)
from grading.grade_gemma import CATEGORY_FIELDS
from grading.spring_2025_prompt import SPRING_2025_RUBRIC


GEMMA_REVIEW_MAX_TOKENS = 7600
LLAMA_REVIEW_MAX_TOKENS = 7600
REVIEW_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class CrossReviewResult:
    validated: ValidatedCrossReview
    raw_text: str
    raw_response: Any
    request_metadata: dict[str, Any]


class CrossReviewFailure(ConsensusValidationError):
    """A cross-review failed while retaining all safe debug evidence."""

    def __init__(self, message: str, debug_metadata: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.debug_metadata = dict(debug_metadata)


def _complete_grading_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "map_file": "<non-empty string>",
        "model": "<non-empty string>",
    }
    for domain, criteria in CATEGORY_FIELDS.items():
        contract[domain] = {
            criterion: {
                "score": "<integer 1-4>",
                "explanation": "<concise string>",
            }
            for criterion in criteria
        }
        contract[domain]["overall_decision"] = "<Yes or No>"
        contract[domain]["if_no_explanation"] = "<string; required when No>"
    contract.update(
        {
            "overall_meets_expectations": "<Yes or No>",
            "strengths": ["<concise string>"],
            "areas_for_improvement": ["<concise string>"],
            "grading_notes": "<string>",
        }
    )
    return contract


def _review_wrapper_contract() -> dict[str, Any]:
    return {
        "reviewer_model": "<gemma or llama>",
        "reviewed_peer_model": "<llama or gemma>",
        "reviewed_grading": _complete_grading_contract(),
        "field_reviews": [
            {
                "path": "<disputed path>",
                "initial_own_value": "<original value>",
                "peer_value": "<peer value>",
                "reviewed_own_value": "<final own value>",
                "action": "<revised|defended|agreed_with_peer|unresolved_uncertainty>",
                "reason": "<one concise sentence>",
            }
        ],
        "changed_field_paths": ["<path>"],
        "defended_field_paths": ["<path>"],
        "review_confidence": "<number 0-1>",
    }


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
    contract = json.dumps(_review_wrapper_contract(), separators=(",", ":"))
    return (
        "OUTPUT CONTRACT (FIRST AND MANDATORY)\n"
        "Your response is invalid unless it contains: (1) one complete reviewed_grading "
        "object, (2) all four rubric domains, (3) all 15 criterion scores and explanations, "
        "(4) map_file, model, overall_meets_expectations, strengths, "
        "areas_for_improvement, and grading_notes, (5) one field_reviews item for every "
        "disputed field, (6) changed_field_paths, (7) defended_field_paths, and "
        "(8) review_confidence. Do not return a partial patch, only disputed fields, or "
        "only field_reviews. Do not omit unchanged fields. Required shape:\n"
        + contract
        + "\n\nBegin from your complete immutable initial grading. Preserve every undisputed "
        "field exactly. Reconsider only the listed disputed fields, except that a "
        "logically related domain or overall decision may be reconsidered. Return the "
        "entire reviewed grading object, including all unchanged fields. The original "
        "image is authoritative; the peer output is critique, not an answer key. Never "
        "average or compromise merely to agree. Revise only when visible image evidence "
        "and the exact rubric support it. Do not infer invisible relationships. Keep each "
        "field_reviews reason to one concise sentence; do not output chain-of-thought, "
        "extended deliberation, or repeated rubric text.\n\n"
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
        + "\n\nReturn raw JSON only. No Markdown or prose outside JSON."
    )


def _attempt_debug(call: ProviderCallResult, raw_text: str) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        parsed = clean_json_object(raw_text)
    except Exception as exc:
        parse_error = str(exc)
    reviewed = parsed.get("reviewed_grading") if isinstance(parsed, dict) else None
    return {
        "provider": call.metadata.get("provider"),
        "model_id": call.metadata.get("model_id"),
        "prompt_character_count": call.metadata.get("prompt_character_count"),
        "max_tokens": call.metadata.get("max_tokens"),
        "image_resent": call.metadata.get("image_resent"),
        "http_status": call.metadata.get("http_status"),
        "finish_reason": call.metadata.get("finish_reason"),
        "completion_tokens": call.metadata.get("completion_tokens"),
        "raw_response": call.raw_response,
        "raw_text": raw_text,
        "parsed_json": parsed,
        "parsed_top_level_keys": sorted(parsed) if isinstance(parsed, dict) else [],
        "reviewed_grading_present": isinstance(reviewed, dict),
        "reviewed_grading_top_level_keys": (
            sorted(reviewed) if isinstance(reviewed, dict) else []
        ),
        "parse_error": parse_error,
        "truncated": call.metadata.get("finish_reason") == "length",
        "validation_object_path": "response.reviewed_grading",
    }


def _looks_substantively_complete(raw_text: str) -> bool:
    required = {
        "reviewer_model",
        "reviewed_peer_model",
        "reviewed_grading",
        "field_reviews",
        "changed_field_paths",
        "defended_field_paths",
        "review_confidence",
        "map_file",
        "model",
        "strengths",
        "areas_for_improvement",
        "grading_notes",
        *CATEGORY_FIELDS.keys(),
    }
    return all(f'"{key}"' in raw_text for key in required)


def _score_sequence(raw_text: str) -> list[int]:
    return [int(value) for value in re.findall(r'"score"\s*:\s*([1-4])\b', raw_text)]


def _repair_semantic_signature(raw_text: str) -> dict[str, list[str] | list[int]]:
    return {
        "scores": _score_sequence(raw_text),
        "domain_decisions": re.findall(
            r'"overall_decision"\s*:\s*"(Yes|No)"', raw_text
        ),
        "final_decisions": re.findall(
            r'"overall_meets_expectations"\s*:\s*"(Yes|No)"', raw_text
        ),
        "review_actions": re.findall(
            r'"action"\s*:\s*"(revised|defended|agreed_with_peer|unresolved_uncertainty)"',
            raw_text,
        ),
    }


def _format_repair_prompt(raw_text: str) -> str:
    return (
        "FORMAT REPAIR ONLY. The cross-review below contains the substantive review but "
        "is not valid JSON. Return the exact same review as one valid raw JSON object. "
        "Do not re-review, invent fields, change any score or decision, change any "
        "field-review action, or omit unchanged grading fields. Required complete shape:\n"
        + json.dumps(_review_wrapper_contract(), separators=(",", ":"))
        + "\n\nMALFORMED CROSS-REVIEW\n"
        + raw_text
    )


def _validate_call(
    call: ProviderCallResult,
    *,
    reviewer_model: str,
    reviewed_peer_model: str,
    initial_own: Mapping[str, Any],
    initial_peer: Mapping[str, Any],
    disagreement_paths: set[str],
) -> ValidatedCrossReview:
    payload = clean_json_object(call.raw_text)
    return validate_cross_review(
        payload,
        reviewer_model=reviewer_model,
        reviewed_peer_model=reviewed_peer_model,
        initial_own=initial_own,
        initial_peer=initial_peer,
        initial_disagreement_paths=disagreement_paths,
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
    progress_callback: Any | None = None,
) -> CrossReviewResult:
    disagreements = list(initial_comparison.get("disagreements", []))
    disagreement_paths = {str(item["path"]) for item in disagreements}
    prompt = build_cross_review_prompt(
        reviewer_model=reviewer_model,
        reviewed_peer_model=reviewed_peer_model,
        initial_own=initial_own,
        initial_peer=initial_peer,
        disagreements=disagreements,
    )
    max_tokens = (
        GEMMA_REVIEW_MAX_TOKENS
        if reviewer_model == "gemma"
        else LLAMA_REVIEW_MAX_TOKENS
    )
    attempts: list[dict[str, Any]] = []
    first = invoke(
        provider=provider,
        model_id=model_id,
        prompt=prompt,
        image_base64=image_base64,
        max_tokens=max_tokens,
        timeout_seconds=REVIEW_TIMEOUT_SECONDS,
        stage="cross_review",
        progress_callback=progress_callback,
    )
    first_debug = _attempt_debug(first, first.raw_text)
    attempts.append(first_debug)
    try:
        validated = _validate_call(
            first,
            reviewer_model=reviewer_model,
            reviewed_peer_model=reviewed_peer_model,
            initial_own=initial_own,
            initial_peer=initial_peer,
            disagreement_paths=disagreement_paths,
        )
        first_debug.update(
            {
                "reviewed_grading_validation_success": True,
                "wrapper_validation_success": True,
                "response_classification": "complete",
            }
        )
        return CrossReviewResult(
            validated,
            first.raw_text,
            first.raw_response,
            {
                **first.metadata,
                "review_call_count": 1,
                "recovery_attempted": False,
                "final_review_status": "complete",
                "attempts": attempts,
            },
        )
    except Exception as first_error:
        first_debug.update(
            {
                "reviewed_grading_validation_success": False,
                "wrapper_validation_success": False,
                "validation_error": str(first_error),
            }
        )

    format_only = first_debug["parse_error"] is not None and _looks_substantively_complete(
        first.raw_text
    )
    first_debug["response_classification"] = (
        "malformed_complete" if format_only else "substantively_incomplete"
    )
    recovery_prompt = (
        _format_repair_prompt(first.raw_text)
        if format_only
        else "CROSS-REVIEW RETRY: Your prior response was substantively incomplete. "
        "Return a complete cross-review; do not return a patch.\n\n" + prompt
    )
    recovered = invoke(
        provider=provider,
        model_id=model_id,
        prompt=recovery_prompt,
        image_base64=None if format_only else image_base64,
        max_tokens=max_tokens,
        timeout_seconds=REVIEW_TIMEOUT_SECONDS,
        stage="cross_review",
        progress_callback=progress_callback,
    )
    recovered_debug = _attempt_debug(recovered, recovered.raw_text)
    recovered_debug["recovery_type"] = "format_only" if format_only else "full_review_retry"
    attempts.append(recovered_debug)
    try:
        validated = _validate_call(
            recovered,
            reviewer_model=reviewer_model,
            reviewed_peer_model=reviewed_peer_model,
            initial_own=initial_own,
            initial_peer=initial_peer,
            disagreement_paths=disagreement_paths,
        )
        if format_only:
            before = _repair_semantic_signature(first.raw_text)
            after = _repair_semantic_signature(recovered.raw_text)
            for field, original_values in before.items():
                if original_values and original_values != after[field]:
                    raise ConsensusValidationError(
                        f"Format-only repair changed model-generated {field}."
                    )
        recovered_debug.update(
            {
                "reviewed_grading_validation_success": True,
                "wrapper_validation_success": True,
                "response_classification": "complete_after_recovery",
            }
        )
        return CrossReviewResult(
            validated,
            recovered.raw_text,
            {"initial": first.raw_response, "recovery": recovered.raw_response},
            {
                **recovered.metadata,
                "review_call_count": 2,
                "recovery_attempted": True,
                "recovery_type": recovered_debug["recovery_type"],
                "final_review_status": "complete",
                "attempts": attempts,
            },
        )
    except Exception as recovery_error:
        recovered_debug.update(
            {
                "reviewed_grading_validation_success": False,
                "wrapper_validation_success": False,
                "validation_error": str(recovery_error),
                "response_classification": "failed_after_recovery",
            }
        )
        raise CrossReviewFailure(
            str(recovery_error),
            {
                "attempted": True,
                "provider": provider,
                "model_id": model_id,
                "prompt_character_count": len(prompt),
                "max_tokens": max_tokens,
                "timeout_seconds": REVIEW_TIMEOUT_SECONDS,
                "streaming_enabled": False,
                "image_resent": True,
                "recovery_attempted": True,
                "final_review_status": "failed",
                "attempts": attempts,
            },
        ) from recovery_error
