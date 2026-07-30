"""Deterministic comparison only; this module never resolves grading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from grading.multimodal_feedback import CRITERION_FIELDS


@dataclass(frozen=True)
class ComparisonField:
    path: str
    label: str
    disagreement_type: str


DOMAIN_LABELS = {
    "knowledge_acquisition": "Knowledge Acquisition",
    "integration": "Integration",
    "application": "Application",
    "transfer": "Transfer",
}


def configured_comparison_fields() -> tuple[ComparisonField, ...]:
    fields: list[ComparisonField] = []
    for domain, criteria in CRITERION_FIELDS.items():
        for criterion in criteria:
            fields.append(
                ComparisonField(
                    path=f"{domain}.{criterion}.score",
                    label=criterion.replace("_", " ").title(),
                    disagreement_type="criterion_score",
                )
            )
        fields.append(
            ComparisonField(
                path=f"{domain}.overall_decision",
                label=f"{DOMAIN_LABELS[domain]} Domain Decision",
                disagreement_type="domain_decision",
            )
        )
    fields.append(
        ComparisonField(
            path="overall_meets_expectations",
            label="Overall Meets Expectations",
            disagreement_type="overall_decision",
        )
    )
    return tuple(fields)


COMPARISON_FIELDS = configured_comparison_fields()
COMPARISON_FIELD_BY_PATH = {field.path: field for field in COMPARISON_FIELDS}


def get_path(data: Mapping[str, Any], path: str) -> Any:
    value: Any = data
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    components = path.split(".")
    target: dict[str, Any] = data
    for component in components[:-1]:
        child = target.get(component)
        if not isinstance(child, dict):
            raise KeyError(path)
        target = child
    target[components[-1]] = value


def compare_gradings(
    gemma: Mapping[str, Any],
    llama: Mapping[str, Any],
) -> dict[str, Any]:
    disagreements: list[dict[str, Any]] = []
    agreement_count = 0
    for field in COMPARISON_FIELDS:
        gemma_value = get_path(gemma, field.path)
        llama_value = get_path(llama, field.path)
        if gemma_value == llama_value:
            agreement_count += 1
            continue
        record: dict[str, Any] = {
            "path": field.path,
            "label": field.label,
            "type": field.disagreement_type,
            "gemma_value": gemma_value,
            "llama_value": llama_value,
        }
        if (
            field.disagreement_type == "criterion_score"
            and isinstance(gemma_value, int)
            and not isinstance(gemma_value, bool)
            and isinstance(llama_value, int)
            and not isinstance(llama_value, bool)
        ):
            record["absolute_difference"] = abs(gemma_value - llama_value)
        disagreements.append(record)
    total = len(COMPARISON_FIELDS)
    return {
        "total_compared_fields": total,
        "agreement_count": agreement_count,
        "disagreement_count": len(disagreements),
        "initial_agreement_rate": agreement_count / total if total else 1.0,
        "disagreements": disagreements,
    }


def classify_post_review(
    initial_comparison: Mapping[str, Any],
    initial_gemma: Mapping[str, Any],
    initial_llama: Mapping[str, Any],
    reviewed_gemma: Mapping[str, Any] | None,
    reviewed_llama: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resolutions: list[dict[str, Any]] = []
    effective_gemma = reviewed_gemma or initial_gemma
    effective_llama = reviewed_llama or initial_llama
    for disagreement in initial_comparison.get("disagreements", []):
        path = str(disagreement["path"])
        initial_g = get_path(initial_gemma, path)
        initial_l = get_path(initial_llama, path)
        reviewed_g = get_path(effective_gemma, path)
        reviewed_l = get_path(effective_llama, path)
        if reviewed_gemma is None or reviewed_llama is None:
            status = "review_unavailable"
        elif reviewed_g == reviewed_l:
            gemma_changed = reviewed_g != initial_g
            llama_changed = reviewed_l != initial_l
            if gemma_changed and llama_changed:
                status = "resolved_by_both_revision"
            elif gemma_changed:
                status = "resolved_by_gemma_revision"
            elif llama_changed:
                status = "resolved_by_llama_revision"
            else:
                status = "resolved_same_value"
        elif reviewed_g == initial_g and reviewed_l == initial_l:
            status = "unresolved_same_as_initial"
        else:
            status = "unresolved_after_revision"
        resolutions.append(
            {
                "path": path,
                "initial_gemma": initial_g,
                "initial_llama": initial_l,
                "reviewed_gemma": reviewed_g,
                "reviewed_llama": reviewed_l,
                "resolution_status": status,
            }
        )
    post_comparison = compare_gradings(effective_gemma, effective_llama)
    return {
        **post_comparison,
        "field_resolutions": resolutions,
        "resolution_status_by_path": {
            item["path"]: item["resolution_status"] for item in resolutions
        },
    }
