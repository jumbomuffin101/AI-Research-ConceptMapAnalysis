"""Orchestrate immutable initial grades, one review round, and adjudication."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from consensus.adjudication import (
    AdjudicationResult,
    ConsensusModelConfig,
    run_adjudication,
)
from consensus.comparison import classify_post_review, compare_gradings
from consensus.cross_review import CrossReviewFailure, CrossReviewResult, run_cross_review
from grading import grade_gemma, grade_llama
from interface.grading_runner import EvaluationResult


CONSENSUS_PIPELINE_VERSION = "single-round-consensus-v2-complete-output"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web_demo"
DEBUG_DIR = OUTPUT_DIR / "debug"


@dataclass(frozen=True)
class ConsensusPipelineResult:
    export: dict[str, Any]
    output_path: Path
    debug_path: Path


def _safe_stem(filename: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_")[:60] or "map"


def _initial_mapping(
    initial_results: Mapping[str, Any] | Iterable[Any],
) -> dict[str, dict[str, Any]]:
    if isinstance(initial_results, Mapping):
        source = dict(initial_results)
    else:
        source = {}
        for item in initial_results:
            if isinstance(item, EvaluationResult):
                source[item.model_name] = item.data
    mapped: dict[str, dict[str, Any]] = {}
    for label, key in (
        ("Gemma", "gemma"),
        ("Llama 3.2 90B Vision", "llama"),
        ("gemma", "gemma"),
        ("llama", "llama"),
    ):
        value = source.get(label)
        if isinstance(value, EvaluationResult):
            value = value.data
        if isinstance(value, dict):
            mapped[key] = copy.deepcopy(value)
    return mapped


def _render_deliberation_images(
    pdf_path: Path,
    debug_prefix: Path,
) -> dict[str, str]:
    gemma_info = grade_gemma.render_pdf_first_page(
        pdf_path,
        Path(f"{debug_prefix}_gemma_deliberation.jpg"),
    )
    llama_info = grade_llama.render_pdf_first_page(
        pdf_path,
        Path(f"{debug_prefix}_llama_deliberation.jpg"),
    )
    return {
        "gemma": gemma_info,
        "llama": str(llama_info["base64"]),
    }


def _unavailable_consensus(reason: str) -> dict[str, Any]:
    return {
        "consensus_status": "unavailable",
        "consensus_grading": None,
        "criterion_resolutions": [],
        "unresolved_disagreements": [],
        "consensus_confidence": None,
        "human_review_recommended": True,
        "consensus_notes": reason,
    }


def _review_debug(
    result: CrossReviewResult | None,
    error: Exception | None,
) -> dict[str, Any]:
    if result is None:
        metadata = getattr(error, "debug_metadata", {}) if error else {}
        return {
            "response": metadata.get("attempts"),
            "validation": {"valid": False, "error": str(error) if error else "not attempted"},
            "request_metadata": metadata,
        }
    return {
        "response": result.raw_response,
        "raw_text": result.raw_text,
        "validation": {
            "valid": True,
            "warnings": list(result.validated.warnings),
            "changed_field_paths": list(result.validated.changed_field_paths),
        },
        "request_metadata": result.request_metadata,
    }


def _review_failure_payload(
    model: str,
    error: Exception | None,
    debug_path: Path,
) -> dict[str, Any] | None:
    if error is None:
        return None
    error_type = (
        "incomplete_output"
        if isinstance(error, CrossReviewFailure)
        else "provider_or_validation_error"
    )
    return {
        "status": "failed",
        "model": model,
        "error_type": error_type,
        "error_message": str(error),
        "raw_debug_path": str(debug_path),
        "initial_result_preserved": True,
    }


def run_consensus_pipeline(
    *,
    pdf_path: Path,
    map_file: str,
    initial_results: Mapping[str, Any] | Iterable[Any],
    consensus_config: ConsensusModelConfig | None = None,
    progress_callback: Any | None = None,
    debug_prefix: Path | None = None,
    image_inputs: Mapping[str, str] | None = None,
    cross_review_runner: Callable[..., CrossReviewResult] = run_cross_review,
    adjudication_runner: Callable[..., AdjudicationResult] = run_adjudication,
) -> ConsensusPipelineResult:
    """Run at most two independent reviews and one model-authored consensus."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex[:8]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    prefix = debug_prefix or (
        DEBUG_DIR / f"{timestamp}_{run_id}_{_safe_stem(map_file)}"
    )
    output_path = OUTPUT_DIR / f"{timestamp}_{run_id}_{_safe_stem(map_file)}_consensus.json"
    debug_path = Path(f"{prefix}_consensus_debug.json")

    initial = _initial_mapping(initial_results)
    initial_gemma = initial.get("gemma")
    initial_llama = initial.get("llama")
    debug: dict[str, Any] = {
        "consensus_pipeline_version": CONSENSUS_PIPELINE_VERSION,
        "initial_gemma_result": initial_gemma,
        "initial_llama_result": initial_llama,
        "gemma_cross_review_attempted": False,
        "gemma_cross_review_image_resent": False,
        "llama_cross_review_attempted": False,
        "llama_cross_review_image_resent": False,
        "consensus_call_attempted": False,
        "consensus_image_resent": False,
    }

    if initial_gemma is None or initial_llama is None:
        missing = "Gemma" if initial_gemma is None else "Llama"
        consensus = _unavailable_consensus(
            f"Two-model consensus is unavailable because {missing} initial grading failed or is missing."
        )
        export = {
            "map_file": map_file,
            "initial_results": {
                "gemma": initial_gemma,
                "llama": initial_llama,
            },
            "initial_comparison": None,
            "cross_reviews": {"gemma": None, "llama": None},
            "post_review_comparison": None,
            "consensus": consensus,
        }
        debug.update(
            {
                "consensus_validation": {"valid": False, "reason": consensus["consensus_notes"]},
                "unresolved_disagreement_count": 0,
                "human_review_recommended": True,
            }
        )
        output_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        return ConsensusPipelineResult(export, output_path, debug_path)

    # These copies remain untouched throughout the service.
    immutable_gemma = copy.deepcopy(initial_gemma)
    immutable_llama = copy.deepcopy(initial_llama)
    initial_comparison = compare_gradings(immutable_gemma, immutable_llama)
    disagreement_paths = [
        str(item["path"]) for item in initial_comparison["disagreements"]
    ]
    debug.update(
        {
            "initial_disagreement_summary": initial_comparison,
            "initial_disagreement_paths": disagreement_paths,
        }
    )

    images = dict(image_inputs or {})
    if not {"gemma", "llama"}.issubset(images):
        images.update(_render_deliberation_images(pdf_path, prefix))

    gemma_review_result: CrossReviewResult | None = None
    llama_review_result: CrossReviewResult | None = None
    gemma_review_error: Exception | None = None
    llama_review_error: Exception | None = None

    if disagreement_paths:
        if progress_callback:
            progress_callback("Gemma is independently reviewing disputed fields")
        debug["gemma_cross_review_attempted"] = True
        debug["gemma_cross_review_image_resent"] = True
        try:
            gemma_review_result = cross_review_runner(
                reviewer_model="gemma",
                reviewed_peer_model="llama",
                provider="OpenRouter",
                model_id=grade_gemma.MODEL,
                image_base64=images["gemma"],
                initial_own=immutable_gemma,
                initial_peer=immutable_llama,
                initial_comparison=initial_comparison,
            )
        except Exception as exc:
            gemma_review_error = exc

        if progress_callback:
            progress_callback("Llama is independently reviewing disputed fields")
        debug["llama_cross_review_attempted"] = True
        debug["llama_cross_review_image_resent"] = True
        try:
            llama_review_result = cross_review_runner(
                reviewer_model="llama",
                reviewed_peer_model="gemma",
                provider="NVIDIA NIM",
                model_id=grade_llama.MODEL,
                image_base64=images["llama"],
                initial_own=immutable_llama,
                initial_peer=immutable_gemma,
                initial_comparison=initial_comparison,
            )
        except Exception as exc:
            llama_review_error = exc

    reviewed_gemma = (
        gemma_review_result.validated.reviewed_grading
        if gemma_review_result
        else None
    )
    reviewed_llama = (
        llama_review_result.validated.reviewed_grading
        if llama_review_result
        else None
    )
    post_review = classify_post_review(
        initial_comparison,
        immutable_gemma,
        immutable_llama,
        reviewed_gemma,
        reviewed_llama,
    )
    gemma_review_payload = (
        gemma_review_result.validated.payload
        if gemma_review_result
        else _review_failure_payload("gemma", gemma_review_error, debug_path)
    )
    llama_review_payload = (
        llama_review_result.validated.payload
        if llama_review_result
        else _review_failure_payload("llama", llama_review_error, debug_path)
    )
    gemma_debug = _review_debug(gemma_review_result, gemma_review_error)
    llama_debug = _review_debug(llama_review_result, llama_review_error)
    debug.update(
        {
            "gemma_cross_review_response": gemma_debug["response"],
            "gemma_cross_review_validation": gemma_debug["validation"],
            "gemma_cross_review_request_metadata": gemma_debug["request_metadata"],
            "gemma_cross_review_final_status": (
                "complete" if gemma_review_result else "failed"
            ),
            "llama_cross_review_response": llama_debug["response"],
            "llama_cross_review_validation": llama_debug["validation"],
            "llama_cross_review_request_metadata": llama_debug["request_metadata"],
            "llama_cross_review_final_status": (
                "complete" if llama_review_result else "failed"
            ),
            "post_review_disagreement_summary": post_review,
            "resolution_status_by_path": post_review["resolution_status_by_path"],
        }
    )

    config = consensus_config or ConsensusModelConfig.from_environment()
    debug["consensus_model_provider"] = config.provider
    debug["consensus_model_id"] = config.model_id
    consensus_result: AdjudicationResult | None = None
    consensus_error: Exception | None = None
    if progress_callback:
        progress_callback("Generating model-authored consensus")
    debug["consensus_call_attempted"] = True
    debug["consensus_image_resent"] = True
    consensus_image = (
        images["llama"]
        if config.provider.lower() in {"nvidia", "nvidia nim", "llama"}
        else images["gemma"]
    )
    try:
        consensus_result = adjudication_runner(
            config=config,
            image_base64=consensus_image,
            initial_gemma=immutable_gemma,
            initial_llama=immutable_llama,
            gemma_review=gemma_review_payload,
            llama_review=llama_review_payload,
            initial_comparison=initial_comparison,
            post_review_comparison=post_review,
        )
        consensus = consensus_result.consensus
    except Exception as exc:
        consensus_error = exc
        consensus = _unavailable_consensus(
            f"Consensus model call failed: {exc}"
        )

    unresolved_count = len(consensus.get("unresolved_disagreements", []))
    gemma_meta = gemma_debug["request_metadata"]
    llama_meta = llama_debug["request_metadata"]
    consensus_meta = (
        consensus_result.request_metadata
        if consensus_result
        else getattr(consensus_error, "debug_metadata", {})
    )
    debug.update(
        {
            "consensus_raw_response": (
                consensus_result.raw_response
                if consensus_result
                else consensus_meta.get("attempts")
            ),
            "consensus_raw_text": (
                consensus_result.raw_text
                if consensus_result
                else [
                    item.get("raw_text")
                    for item in consensus_meta.get("attempts", [])
                    if isinstance(item, Mapping)
                ]
            ),
            "consensus_validation": (
                {
                    "valid": True,
                    "consensus_grading_present": True,
                    "wrapper_validation_success": True,
                }
                if consensus_result
                else {
                    "valid": False,
                    "error": str(consensus_error),
                    "details": getattr(consensus_error, "debug_metadata", {}),
                }
            ),
            "consensus_request_metadata": consensus_meta,
            "unresolved_disagreement_count": unresolved_count,
            "human_review_recommended": bool(
                consensus.get("human_review_recommended")
            ),
            "request_timeout_by_stage": {
                "gemma_cross_review": (
                    gemma_meta.get("timeout_seconds")
                ),
                "llama_cross_review": (
                    llama_meta.get("timeout_seconds")
                ),
                "consensus": (
                    consensus_meta.get("timeout_seconds")
                ),
            },
            "prompt_size_by_stage": {
                "gemma_cross_review": (
                    gemma_meta.get("prompt_character_count")
                ),
                "llama_cross_review": (
                    llama_meta.get("prompt_character_count")
                ),
                "consensus": (
                    consensus_meta.get("prompt_character_count")
                ),
            },
            "max_tokens_by_stage": {
                "gemma_cross_review": (
                    gemma_meta.get("max_tokens")
                ),
                "llama_cross_review": (
                    llama_meta.get("max_tokens")
                ),
                "consensus": (
                    consensus_meta.get("max_tokens")
                ),
            },
            "streaming_configuration_by_stage": {
                "gemma_cross_review": (
                    gemma_meta.get("streaming_enabled")
                ),
                "llama_cross_review": (
                    llama_meta.get("streaming_enabled")
                ),
                "consensus": (
                    consensus_meta.get("streaming_enabled")
                ),
            },
        }
    )
    export = {
        "map_file": map_file,
        "initial_results": {
            "gemma": immutable_gemma,
            "llama": immutable_llama,
        },
        "initial_comparison": initial_comparison,
        "cross_reviews": {
            "gemma": gemma_review_payload,
            "llama": llama_review_payload,
        },
        "post_review_comparison": post_review,
        "consensus": consensus,
    }
    output_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")

    if initial_gemma != immutable_gemma or initial_llama != immutable_llama:
        raise RuntimeError("Consensus pipeline mutated immutable initial grading.")
    return ConsensusPipelineResult(export, output_path, debug_path)
