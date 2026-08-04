"""One model-generated final adjudication after independent cross-review."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from consensus.comparison import COMPARISON_FIELDS, get_path
from consensus.providers import ProviderCallResult, invoke_model
from consensus.schemas import clean_json_object, validate_consensus
from grading import grade_gemma, grade_llama
from grading.spring_2025_prompt import SPRING_2025_RUBRIC


CONSENSUS_MAX_TOKENS = 6000
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


def _grading_projection(grading: Mapping[str, Any] | None) -> Any:
    if grading is None:
        return {"status": "review_unavailable"}
    return {
        "model": grading.get("model"),
        "values": {
            field.path: get_path(grading, field.path) for field in COMPARISON_FIELDS
        },
        "explanations": {
            field.path.removesuffix(".score"): get_path(
                grading,
                field.path.removesuffix(".score") + ".explanation",
            )
            for field in COMPARISON_FIELDS
            if field.disagreement_type == "criterion_score"
        },
    }


def build_consensus_prompt(
    *,
    initial_gemma: Mapping[str, Any],
    initial_llama: Mapping[str, Any],
    gemma_review: Mapping[str, Any] | None,
    llama_review: Mapping[str, Any] | None,
    initial_comparison: Mapping[str, Any],
    post_review_comparison: Mapping[str, Any],
) -> str:
    context = {
        "initial_gemma": _grading_projection(initial_gemma),
        "initial_llama": _grading_projection(initial_llama),
        "gemma_review": gemma_review,
        "llama_review": llama_review,
        "initial_disagreements": initial_comparison.get("disagreements", []),
        "post_review_comparison": post_review_comparison,
    }
    return (
        "Generate the final model-authored consensus for two independent medical concept-map "
        "graders. Inspect the original concept-map image; it is authoritative. Peer outputs "
        "and defenses are evidence, not ground truth. Do not average, choose the higher or "
        "lower score automatically, compromise to a middle value, prefer your own provider, "
        "or force agreement. Review each disputed criterion independently using the exact "
        "rubric. Distinguish missing detail from missing reasoning. Do not infer invisible "
        "relationships. Score 4 does not require encyclopedic completeness; 3 requires "
        "substantial evidence with a meaningful limitation; 2 requires a genuinely "
        "underdeveloped central relationship. Domain decisions are holistic but require "
        "evidence of their central objective. Explicitly retain ambiguity and recommend "
        "human review rather than invent certainty. Silently check consistency before output.\n\n"
        + SPRING_2025_RUBRIC
        + "\n\nDELIBERATION CONTEXT\n"
        + json.dumps(context, separators=(",", ":"), ensure_ascii=False)
        + "\n\nReturn one raw JSON object containing consensus_status, consensus_grading "
        "(the complete compact grading schema), "
        "criterion_resolutions, unresolved_disagreements, consensus_confidence, "
        "human_review_recommended, and consensus_notes. For every initially disputed path, "
        "include its initial and reviewed values, the model-generated consensus_value, "
        "status, resolution_basis, human_review_recommended, and confidence. If a disagreement "
        "remains, still provide your best image-and-rubric-based value in consensus_grading "
        "but also list the field under unresolved_disagreements. Any unresolved field requires "
        "consensus_status=complete_with_human_review and human_review_recommended=true. "
        "No Markdown or prose outside JSON."
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
        initial_gemma=initial_gemma,
        initial_llama=initial_llama,
        gemma_review=gemma_review,
        llama_review=llama_review,
        initial_comparison=initial_comparison,
        post_review_comparison=post_review_comparison,
    )
    call = invoke(
        provider=config.provider,
        model_id=config.model_id,
        prompt=prompt,
        image_base64=image_base64,
        max_tokens=CONSENSUS_MAX_TOKENS,
        timeout_seconds=CONSENSUS_TIMEOUT_SECONDS,
    )
    payload = clean_json_object(call.raw_text)
    validated = validate_consensus(
        payload,
        initial_disagreement_paths={
            str(item["path"])
            for item in initial_comparison.get("disagreements", [])
        },
    )
    return AdjudicationResult(
        validated,
        call.raw_text,
        call.raw_response,
        {**call.metadata, "image_resent": True, "consensus_call_count": 1},
    )
