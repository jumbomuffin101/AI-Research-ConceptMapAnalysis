"""One model-generated final adjudication after independent cross-review."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from consensus.cross_review import _complete_grading_contract
from consensus.comparison import COMPARISON_FIELDS, get_path
from consensus.providers import ProviderCallResult, invoke_model
from consensus.schemas import (
    ConsensusValidationError,
    ConsensusResolutionConsistencyError,
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
    canonical_paths = [field.path for field in COMPARISON_FIELDS]
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
        "reasons concise and output no chain-of-thought. Use this generation order: "
        "(1) review the image and grader outputs, (2) produce the complete "
        "consensus_grading, (3) freeze consensus_grading, (4) produce "
        "criterion_resolutions by looking up values in that frozen grading, (5) produce "
        "unresolved_disagreements, and (6) run the consistency check below. Do not "
        "independently adjudicate a field again while writing its resolution.\n\n"
        + SPRING_2025_RUBRIC
        + "\n\nDELIBERATION CONTEXT\n"
        + json.dumps(context, separators=(",", ":"), ensure_ascii=False)
        + "\n\nCANONICAL COMPARISON PATHS\n"
        + json.dumps(canonical_paths, separators=(",", ":"))
        + "\nReturn exactly one criterion_resolutions item for every canonical path above. "
        "Use each path verbatim. Do not use aliases, omit paths, duplicate paths, or add "
        "invented paths. The value in each criterion_resolutions[].consensus_value must "
        "exactly equal the corresponding value in consensus_grading at the listed path. "
        "Treat consensus_grading as the authoritative final adjudication within your "
        "response. The resolution_basis summarizes the already-made decision; it must not "
        "generate a second value. "
        "If unresolved, provide the best image-and-rubric-based value in consensus_grading, "
        "also list it under unresolved_disagreements, use "
        "consensus_status=complete_with_human_review, and recommend human review. An "
        "unresolved status may remain even though consensus_value contains the best final "
        "adjudication.\n\nCONSISTENCY CHECK BEFORE RESPONDING\n"
        "For every criterion_resolutions item: (1) read its path, (2) look up the final "
        "value at that exact path inside consensus_grading, (3) copy that exact value into "
        "consensus_value, (4) verify the types match, and (5) do not return JSON until all "
        "entries match. Criterion score paths must contain integers 1 through 4. Domain "
        "decision paths and overall_meets_expectations must contain exactly Yes or No. "
        "Return raw JSON only, without Markdown or surrounding prose."
    )


def _resolution_diagnostics(
    parsed: Mapping[str, Any] | None,
    required_paths: set[str],
) -> dict[str, Any]:
    resolutions = parsed.get("criterion_resolutions") if isinstance(parsed, Mapping) else None
    grading = parsed.get("consensus_grading") if isinstance(parsed, Mapping) else None
    if not isinstance(resolutions, list):
        resolutions = []
    paths = [str(item.get("path", "")) for item in resolutions if isinstance(item, Mapping)]
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    observed = set(paths)
    mismatches: list[dict[str, Any]] = []
    if isinstance(grading, Mapping):
        for item in resolutions:
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path", ""))
            if path not in required_paths:
                continue
            grading_value = get_path(grading, path)
            resolution_value = item.get("consensus_value")
            if type(resolution_value) is not type(grading_value) or resolution_value != grading_value:
                mismatches.append(
                    {
                        "path": path,
                        "grading_value": grading_value,
                        "resolution_value": resolution_value,
                        "grading_type": type(grading_value).__name__,
                        "resolution_type": type(resolution_value).__name__,
                    }
                )
    return {
        "consensus_resolution_consistency_checked": isinstance(grading, Mapping),
        "compared_resolution_count": len(paths),
        "duplicate_resolution_paths": duplicates,
        "missing_resolution_paths": sorted(required_paths - observed),
        "extra_resolution_paths": sorted(observed - required_paths),
        "consensus_value_mismatches": mismatches,
    }


def _attempt_debug(
    call: ProviderCallResult,
    required_paths: set[str],
) -> dict[str, Any]:
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
        **_resolution_diagnostics(parsed, required_paths),
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


def _consistency_repair_prompt(
    raw_text: str,
    canonical_paths: list[str],
    mismatches: list[dict[str, Any]],
) -> str:
    return (
        "CONSISTENCY REPAIR ONLY. The JSON below contains a complete model-authored "
        "consensus, but some criterion_resolutions consensus_value fields disagree with "
        "the authoritative consensus_grading. Return one valid raw JSON object. Preserve "
        "consensus_grading byte-for-value: do not change any score, domain decision, final "
        "decision, explanation, strength, area for improvement, grading note, or other "
        "consensus_grading content. Preserve resolution statuses, resolution bases, "
        "confidence values, human-review flags, and unresolved_disagreements. Update only "
        "the listed mismatched criterion_resolutions[].consensus_value fields by copying "
        "the value at the exact path from consensus_grading. Do not re-adjudicate.\n\n"
        "CANONICAL PATHS\n"
        + json.dumps(canonical_paths, separators=(",", ":"))
        + "\n\nMISMATCHES\n"
        + json.dumps(mismatches, separators=(",", ":"), ensure_ascii=False)
        + "\n\nCOMPLETE ORIGINAL CONSENSUS\n"
        + raw_text
    )


def _without_repairable_consensus_values(
    payload: Mapping[str, Any],
    mismatch_paths: set[str],
) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    resolutions = value.get("criterion_resolutions")
    if isinstance(resolutions, list):
        for item in resolutions:
            if isinstance(item, dict) and str(item.get("path", "")) in mismatch_paths:
                item["consensus_value"] = "<repairable-consensus-value>"
    return value


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
    progress_callback: Any | None = None,
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
        stage="consensus",
        progress_callback=progress_callback,
    )
    disagreement_paths = {
        str(item["path"]) for item in initial_comparison.get("disagreements", [])
    }
    canonical_paths = [field.path for field in COMPARISON_FIELDS]
    required_resolution_paths = set(canonical_paths)
    first_debug = _attempt_debug(first, required_resolution_paths)
    attempts.append(first_debug)
    expected_map_file = str(
        initial_gemma.get("map_file") or initial_llama.get("map_file") or ""
    )
    expected_model = f"consensus:{config.model_id}"
    validation_error: Exception | None = None
    try:
        payload = clean_json_object(first.raw_text)
        validated = validate_consensus(
            payload,
            initial_disagreement_paths=disagreement_paths,
            required_resolution_paths=required_resolution_paths,
            expected_map_file=expected_map_file,
            expected_model=expected_model,
        )
        first_debug.update(
            {
                "consensus_grading_validation_success": True,
                "wrapper_validation_success": True,
                "response_classification": "complete",
                "consistency_repair_required": False,
                "consistency_repair_attempted": False,
                "consistency_repair_success": False,
                "consensus_grading_preserved_during_repair": None,
                "final_consistency_validation_success": True,
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
        validation_error = first_error
        first_debug.update(
            {
                "consensus_grading_validation_success": False,
                "wrapper_validation_success": False,
                "validation_error": str(first_error),
            }
        )

    if isinstance(validation_error, ConsensusResolutionConsistencyError):
        first_debug.update(
            {
                "response_classification": "resolution_consistency_mismatch",
                "consistency_repair_required": True,
                "consistency_repair_attempted": True,
                "consistency_repair_success": False,
                "final_consistency_validation_success": False,
                "consensus_value_mismatches": validation_error.mismatches,
            }
        )
        original_payload = clean_json_object(first.raw_text)
        repair = invoke(
            provider=config.provider,
            model_id=config.model_id,
            prompt=_consistency_repair_prompt(
                first.raw_text,
                canonical_paths,
                validation_error.mismatches,
            ),
            image_base64=None,
            max_tokens=CONSENSUS_MAX_TOKENS,
            timeout_seconds=CONSENSUS_TIMEOUT_SECONDS,
            stage="consensus",
            progress_callback=progress_callback,
        )
        repair_debug = _attempt_debug(repair, required_resolution_paths)
        repair_debug.update(
            {
                "recovery_type": "consistency_only",
                "consistency_repair_required": True,
                "consistency_repair_attempted": True,
            }
        )
        attempts.append(repair_debug)
        try:
            repaired_payload = clean_json_object(repair.raw_text)
            original_grading = original_payload.get("consensus_grading")
            repaired_grading = repaired_payload.get("consensus_grading")
            grading_preserved = repaired_grading == original_grading
            mismatch_paths = {
                str(item["path"]) for item in validation_error.mismatches
            }
            only_allowed_values_changed = (
                _without_repairable_consensus_values(original_payload, mismatch_paths)
                == _without_repairable_consensus_values(repaired_payload, mismatch_paths)
            )
            repair_debug["consensus_grading_preserved_during_repair"] = grading_preserved
            if not grading_preserved:
                raise ConsensusValidationError(
                    "Consistency repair changed consensus_grading."
                )
            if not only_allowed_values_changed:
                raise ConsensusValidationError(
                    "Consistency repair changed fields other than mismatched consensus_value entries."
                )
            validated = validate_consensus(
                repaired_payload,
                initial_disagreement_paths=disagreement_paths,
                required_resolution_paths=required_resolution_paths,
                expected_map_file=expected_map_file,
                expected_model=expected_model,
            )
            repair_debug.update(
                {
                    "consensus_grading_validation_success": True,
                    "wrapper_validation_success": True,
                    "response_classification": "complete_after_consistency_repair",
                    "consistency_repair_success": True,
                    "final_consistency_validation_success": True,
                }
            )
            return AdjudicationResult(
                validated,
                repair.raw_text,
                {
                    "initial": first.raw_response,
                    "consistency_repair": repair.raw_response,
                },
                {
                    **repair.metadata,
                    "consensus_call_count": 2,
                    "recovery_attempted": True,
                    "recovery_type": "consistency_only",
                    "consistency_repair_required": True,
                    "consistency_repair_attempted": True,
                    "consistency_repair_success": True,
                    "consensus_grading_preserved_during_repair": True,
                    "final_consistency_validation_success": True,
                    "attempts": attempts,
                    "final_status": "complete",
                },
            )
        except Exception as repair_error:
            repair_debug.update(
                {
                    "consensus_grading_validation_success": False,
                    "wrapper_validation_success": False,
                    "validation_error": str(repair_error),
                    "response_classification": "failed_consistency_repair",
                    "consistency_repair_success": False,
                    "final_consistency_validation_success": False,
                }
            )
            raise AdjudicationFailure(
                str(repair_error),
                {
                    "provider": config.provider,
                    "model_id": config.model_id,
                    "prompt_character_count": len(prompt),
                    "max_tokens": CONSENSUS_MAX_TOKENS,
                    "timeout_seconds": CONSENSUS_TIMEOUT_SECONDS,
                    "streaming_enabled": False,
                    "image_resent": True,
                    "consistency_repair_required": True,
                    "consistency_repair_attempted": True,
                    "consistency_repair_success": False,
                    "consensus_grading_preserved_during_repair": repair_debug.get(
                        "consensus_grading_preserved_during_repair"
                    ),
                    "final_consistency_validation_success": False,
                    "attempts": attempts,
                    "final_status": "failed",
                },
            ) from repair_error

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
        stage="consensus",
        progress_callback=progress_callback,
    )
    recovered_debug = _attempt_debug(recovered, required_resolution_paths)
    recovered_debug["recovery_type"] = "format_only" if format_only else "full_consensus_retry"
    attempts.append(recovered_debug)
    try:
        payload = clean_json_object(recovered.raw_text)
        validated = validate_consensus(
            payload,
            initial_disagreement_paths=disagreement_paths,
            required_resolution_paths=required_resolution_paths,
            expected_map_file=expected_map_file,
            expected_model=expected_model,
        )
        recovered_debug.update(
            {
                "consensus_grading_validation_success": True,
                "wrapper_validation_success": True,
                "response_classification": "complete_after_recovery",
                "final_consistency_validation_success": True,
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
                "final_consistency_validation_success": True,
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
                "final_consistency_validation_success": False,
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
                "final_consistency_validation_success": False,
                "final_status": "failed",
            },
        ) from recovery_error
