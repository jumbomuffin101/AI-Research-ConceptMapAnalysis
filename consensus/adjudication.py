"""One model-generated final adjudication after independent cross-review."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from consensus.cross_review import _complete_grading_contract
from consensus.providers import ProviderCallResult, invoke_model
from consensus.schemas import (
    ConsensusValidationError,
    clean_json_object,
    validate_consensus,
)
from grading import grade_gemma, grade_llama
from grading.spring_2025_prompt import SPRING_2025_RUBRIC


CONSENSUS_MAX_TOKENS = 7600
CONSENSUS_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class ConsensusModelConfig:
    provider: str
    model_id: str

    @classmethod
    def from_environment(cls) -> "ConsensusModelConfig":
        provider = os.getenv("CONSENSUS_MODEL_PROVIDER", "OpenRouter").strip()
        default_model = (
            grade_llama.MODEL
            if provider.lower() in {"nvidia", "nvidia nim", "llama"}
            else grade_gemma.MODEL
        )
        return cls(
            provider=provider,
            model_id=os.getenv("CONSENSUS_MODEL_ID", default_model).strip(),
        )


@dataclass(frozen=True)
class AdjudicationResult:
    consensus: dict[str, Any]
    raw_text: str
    raw_response: Any
    request_metadata: dict[str, Any]


class AdjudicationFailure(ConsensusValidationError):
    """Adjudication failed while preserving both provider attempts."""

    def __init__(self, message: str, debug_metadata: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.debug_metadata = dict(debug_metadata)


def _review_context(review: Mapping[str, Any] | None, model: str) -> dict[str, Any]:
    if review is None:
        return {"status": "unavailable", "model": model}
    if review.get("status") == "failed":
        return dict(review)
    return {"status": "available", "review": dict(review)}


def _consensus_contract() -> dict[str, Any]:
    return {
        "consensus_status": "<complete or complete_with_human_review>",
        "consensus_grading": _complete_grading_contract(),
        "criterion_resolutions": [
            {
                "path": "<initially disputed path>",
                "initial_gemma": "<value>",
                "initial_llama": "<value>",
                "reviewed_gemma": "<value>",
                "reviewed_llama": "<value>",
                "consensus_value": "<model-selected value>",
                "status": "<resolved or unresolved>",
                "resolution_basis": "<one concise sentence>",
                "human_review_recommended": "<boolean>",
                "confidence": "<number 0-1>",
            }
        ],
        "unresolved_disagreements": [
            {
                "path": "<path>",
                "status": "unresolved",
                "reason": "<one concise sentence>",
                "human_review_recommended": True,
            }
        ],
        "consensus_confidence": "<number 0-1>",
        "human_review_recommended": "<boolean>",
        "consensus_notes": "<concise string>",
    }


def build_consensus_prompt(
    *,
    config: ConsensusModelConfig,
    initial_gemma: Mapping[str, Any],
    initial_llama: Mapping[str, Any],
    gemma_review: Mapping[str, Any] | None,
    llama_review: Mapping[str, Any] | None,
    initial_comparison: Mapping[str, Any],
    post_review_comparison: Mapping[str, Any],
) -> str:
    map_file = str(initial_gemma.get("map_file") or initial_llama.get("map_file") or "")
    context = {
        "initial_gemma": dict(initial_gemma),
        "initial_llama": dict(initial_llama),
        "gemma_review": _review_context(gemma_review, "gemma"),
        "llama_review": _review_context(llama_review, "llama"),
        "initial_disagreements": initial_comparison.get("disagreements", []),
        "post_review_comparison": post_review_comparison,
    }
    return (
        "OUTPUT CONTRACT (FIRST AND MANDATORY)\n"
        "Your response is invalid unless consensus_grading is a complete compact production "
        "grading containing map_file, model, all four domains, all 15 scores and "
        "explanations, all four domain decisions and if_no_explanation fields, final "
        "overall decision, strengths, areas_for_improvement, and grading_notes. Do not "
        "return only resolutions, only disputed fields, or a partial grading. Required shape:\n"
        + json.dumps(_consensus_contract(), separators=(",", ":"))
        + "\n\nSet consensus_grading.map_file to "
        + json.dumps(map_file)
        + " and consensus_grading.model to "
        + json.dumps(f"consensus:{config.model_id}")
        + ". You must generate all substantive scores, decisions, and narrative fields. "
        "These two known strings are metadata, not Python-authored grading.\n\n"
        "Generate the final model-authored consensus for two independent medical concept-map "
        "graders. Inspect the original concept-map image; it is authoritative. Peer outputs "
        "and defenses are evidence, not ground truth. Do not average, choose a higher or "
        "lower score automatically, or compromise to a middle value. Review every disputed "
        "field independently with the exact rubric and preserve uncertainty when warranted. "
        "If either review status is unavailable, adjudicate from the original image, both "
        "complete initial grades, and the disagreement list; unavailable reviews reduce "
        "confidence but do not permit an incomplete consensus_grading. Keep resolution "
        "reasons concise and output no chain-of-thought.\n\n"
        + SPRING_2025_RUBRIC
        + "\n\nDELIBERATION CONTEXT\n"
        + json.dumps(context, separators=(",", ":"), ensure_ascii=False)
        + "\n\nFor every initially disputed path include exactly one criterion resolution. "
        "If unresolved, provide the best image-and-rubric-based value in consensus_grading, "
        "also list it under unresolved_disagreements, use "
        "consensus_status=complete_with_human_review, and recommend human review. Return raw "
        "JSON only, without Markdown or surrounding prose."
    )


def _attempt_debug(call: ProviderCallResult) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    try:
        parsed = clean_json_object(call.raw_text)
    except Exception as exc:
        parse_error = str(exc)
    grading = parsed.get("consensus_grading") if isinstance(parsed, dict) else None
    return {
        **call.metadata,
        "raw_response": call.raw_response,
        "raw_text": call.raw_text,
        "parsed_json": parsed,
        "parsed_top_level_keys": sorted(parsed) if isinstance(parsed, dict) else [],
        "consensus_grading_present": isinstance(grading, dict),
        "consensus_grading_top_level_keys": (
            sorted(grading) if isinstance(grading, dict) else []
        ),
        "parse_error": parse_error,
        "truncated": call.metadata.get("finish_reason") == "length",
        "validation_object_path": "response.consensus_grading",
    }


def _looks_complete(raw_text: str) -> bool:
    required = {
        "consensus_status",
        "consensus_grading",
        "criterion_resolutions",
        "unresolved_disagreements",
        "consensus_confidence",
        "human_review_recommended",
        "consensus_notes",
        "map_file",
        "model",
        "strengths",
        "areas_for_improvement",
        "grading_notes",
    }
    return all(f'"{key}"' in raw_text for key in required)


def _repair_prompt(raw_text: str) -> str:
    return (
        "FORMAT REPAIR ONLY. Return the exact same consensus as one valid raw JSON object. "
        "Do not re-adjudicate, change scores or decisions, invent missing substantive data, "
        "or omit unchanged fields. Required shape:\n"
        + json.dumps(_consensus_contract(), separators=(",", ":"))
        + "\n\nMALFORMED CONSENSUS\n"
        + raw_text
    )


def run_adjudication(
    *,
    config: ConsensusModelConfig,
    image_base64: str,
    initial_gemma: Mapping[str, Any],
    initial_llama: Mapping[str, Any],
    gemma_review: Mapping[str, Any] | None,
    llama_review: Mapping[str, Any] | None,
    initial_comparison: Mapping[str, Any],
    post_review_comparison: Mapping[str, Any],
    invoke: Callable[..., ProviderCallResult] = invoke_model,
) -> AdjudicationResult:
    prompt = build_consensus_prompt(
        config=config,
        initial_gemma=initial_gemma,
        initial_llama=initial_llama,
        gemma_review=gemma_review,
        llama_review=llama_review,
        initial_comparison=initial_comparison,
        post_review_comparison=post_review_comparison,
    )
    attempts: list[dict[str, Any]] = []
    first = invoke(
        provider=config.provider,
        model_id=config.model_id,
        prompt=prompt,
        image_base64=image_base64,
        max_tokens=CONSENSUS_MAX_TOKENS,
        timeout_seconds=CONSENSUS_TIMEOUT_SECONDS,
    )
    first_debug = _attempt_debug(first)
    attempts.append(first_debug)
    disagreement_paths = {
        str(item["path"]) for item in initial_comparison.get("disagreements", [])
    }
    expected_map_file = str(
        initial_gemma.get("map_file") or initial_llama.get("map_file") or ""
    )
    expected_model = f"consensus:{config.model_id}"
    try:
        payload = clean_json_object(first.raw_text)
        validated = validate_consensus(
            payload,
            initial_disagreement_paths=disagreement_paths,
            expected_map_file=expected_map_file,
            expected_model=expected_model,
        )
        first_debug.update(
            {
                "consensus_grading_validation_success": True,
                "wrapper_validation_success": True,
                "response_classification": "complete",
            }
        )
        return AdjudicationResult(
            validated,
            first.raw_text,
            first.raw_response,
            {
                **first.metadata,
                "consensus_call_count": 1,
                "recovery_attempted": False,
                "final_status": "complete",
                "attempts": attempts,
            },
        )
    except Exception as first_error:
        first_debug.update(
            {
                "consensus_grading_validation_success": False,
                "wrapper_validation_success": False,
                "validation_error": str(first_error),
            }
        )

    format_only = first_debug["parse_error"] is not None and _looks_complete(first.raw_text)
    first_debug["response_classification"] = (
        "malformed_complete" if format_only else "substantively_incomplete"
    )
    recovery_prompt = (
        _repair_prompt(first.raw_text)
        if format_only
        else "CONSENSUS RETRY: The prior response was substantively incomplete. Return the "
        "complete consensus wrapper and complete consensus_grading.\n\n" + prompt
    )
    recovered = invoke(
        provider=config.provider,
        model_id=config.model_id,
        prompt=recovery_prompt,
        image_base64=None if format_only else image_base64,
        max_tokens=CONSENSUS_MAX_TOKENS,
        timeout_seconds=CONSENSUS_TIMEOUT_SECONDS,
    )
    recovered_debug = _attempt_debug(recovered)
    recovered_debug["recovery_type"] = "format_only" if format_only else "full_consensus_retry"
    attempts.append(recovered_debug)
    try:
        payload = clean_json_object(recovered.raw_text)
        validated = validate_consensus(
            payload,
            initial_disagreement_paths=disagreement_paths,
            expected_map_file=expected_map_file,
            expected_model=expected_model,
        )
        recovered_debug.update(
            {
                "consensus_grading_validation_success": True,
                "wrapper_validation_success": True,
                "response_classification": "complete_after_recovery",
            }
        )
        return AdjudicationResult(
            validated,
            recovered.raw_text,
            {"initial": first.raw_response, "recovery": recovered.raw_response},
            {
                **recovered.metadata,
                "consensus_call_count": 2,
                "recovery_attempted": True,
                "recovery_type": recovered_debug["recovery_type"],
                "final_status": "complete",
                "attempts": attempts,
            },
        )
    except Exception as recovery_error:
        recovered_debug.update(
            {
                "consensus_grading_validation_success": False,
                "wrapper_validation_success": False,
                "validation_error": str(recovery_error),
                "response_classification": "failed_after_recovery",
            }
        )
        raise AdjudicationFailure(
            str(recovery_error),
            {
                "provider": config.provider,
                "model_id": config.model_id,
                "prompt_character_count": len(prompt),
                "max_tokens": CONSENSUS_MAX_TOKENS,
                "timeout_seconds": CONSENSUS_TIMEOUT_SECONDS,
                "streaming_enabled": False,
                "image_resent": True,
                "review_availability_statuses": {
                    "gemma": _review_context(gemma_review, "gemma")["status"],
                    "llama": _review_context(llama_review, "llama")["status"],
                },
                "attempts": attempts,
                "final_status": "failed",
            },
        ) from recovery_error
