"""Shared contract and validation for model-generated multimodal feedback.

This module never assigns or changes rubric scores or decisions. It validates
visual evidence separately so a malformed overlay cannot corrupt valid grading.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping


CRITERION_FIELDS = {
    "knowledge_acquisition": [
        "basic_science",
        "health_system_science",
        "clinical_science",
        "patient_case_information",
        "determinants_of_health",
    ],
    "integration": [
        "prioritized_differential_diagnosis",
        "illness_scripts",
        "basic_to_foundational_science",
        "patient_data_to_clinical_information",
        "patient_data_to_basic_science",
    ],
    "application": [
        "working_diagnosis_pathophysiology",
        "patient_data_pathophysiology",
    ],
    "transfer": [
        "prior_basic_science",
        "prior_clinical_concepts",
        "deepens_understanding",
    ],
}

IMPORTANCE_VALUES = {"minor", "moderate", "major"}
MULTIMODAL_SCHEMA_VERSION = "grounded-feedback-v1"
CRITERION_EVIDENCE_FIELDS = (
    "supporting_evidence",
    "missing_evidence",
    "criterion_confidence",
    "human_review_recommended",
)
DOMAIN_EVIDENCE_FIELDS = ("visual_summary",)
TOP_LEVEL_EVIDENCE_FIELDS = ("multimodal_feedback", "learning_feedback")


@dataclass(frozen=True)
class MultimodalValidation:
    available: bool
    complete: bool
    warnings: tuple[str, ...]
    missing_fields: tuple[str, ...]
    invalid_bbox_count: int


def evidence_schema_fragment() -> dict[str, Any]:
    """Canonical output fragment with non-semantic placeholders."""
    return {
        "supporting_evidence": [
            {
                "evidence_text": "<visible criterion-specific evidence>",
                "location_description": "<where it appears>",
                "bbox": ["<x_min 0..1>", "<y_min 0..1>", "<x_max 0..1>", "<y_max 0..1>"],
                "relationship_type": "<visible relationship type>",
                "confidence": "<number 0..1>",
            }
        ],
        "missing_evidence": [
            {
                "missing_relationship": "<important missing relationship>",
                "suggested_connection": "<specific educational connection>",
                "importance": "<minor, moderate, or major>",
            }
        ],
        "criterion_confidence": "<number 0..1>",
        "human_review_recommended": "<boolean>",
    }


def extend_grading_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a prompt template extended with grounded-feedback fields."""
    extended = copy.deepcopy(dict(schema))
    for domain, criteria in CRITERION_FIELDS.items():
        section = extended.get(domain)
        if not isinstance(section, dict):
            continue
        for criterion in criteria:
            item = section.get(criterion)
            if isinstance(item, dict):
                item.update(copy.deepcopy(evidence_schema_fragment()))
        section["visual_summary"] = {
            "strongest_visible_evidence": ["<concise visible evidence>"],
            "most_important_missing_connection": "<concise missing connection or empty string>",
            "domain_confidence": "<number 0..1>",
            "human_review_recommended": "<boolean>",
        }
    extended["multimodal_feedback"] = {
        "strongest_regions": [
            {
                "description": "<educationally important visible region>",
                "bbox": ["<x_min 0..1>", "<y_min 0..1>", "<x_max 0..1>", "<y_max 0..1>"],
                "confidence": "<number 0..1>",
            }
        ],
        "highest_priority_improvements": [
            {
                "current_state": "<visible current state>",
                "missing_bridge": "<missing conceptual bridge>",
                "suggested_revision": "<specific educational revision>",
                "bbox": None,
                "importance": "<minor, moderate, or major>",
            }
        ],
        "overall_visual_confidence": "<number 0..1>",
        "human_review_recommended": "<boolean>",
    }
    extended["learning_feedback"] = [
        {
            "criterion": "<criterion key>",
            "observed_evidence": "<visible evidence>",
            "guiding_question": "<question that promotes learner reasoning>",
            "hint": "<brief hint without giving the complete answer>",
            "bbox": None,
            "confidence": "<number 0..1>",
        }
    ]
    return extended


def compact_grounding_instructions() -> str:
    return (
        "\nGROUNDED MULTIMODAL FEEDBACK\n"
        "For every criterion, return supporting_evidence (0-3 strongest visible items), "
        "missing_evidence, criterion_confidence (0..1), and human_review_recommended. "
        "Each supporting item needs evidence_text, location_description, bbox, "
        "relationship_type, and confidence. bbox is null when localization is uncertain; "
        "otherwise use normalized [x_min,y_min,x_max,y_max] coordinates relative to the "
        "original image, top-left origin, with 0<=values<=1 and increasing bounds. Never "
        "invent a box. For scores below 4, identify a specific missing relationship when "
        "meaningful and label importance minor, moderate, or major. Add each domain's "
        "visual_summary and the top-level multimodal_feedback (at most 3 strongest_regions "
        "and 3 highest_priority_improvements). Also return learning_feedback with grounded "
        "guiding questions and hints; do not change the grading result for Learning Mode.\n"
        "Evidence must be visibly demonstrated. Do not infer relationships from proximity, "
        "uninterpreted arrows, a diagnosis alone, symptoms beside a diagnosis, a named "
        "mechanism, or terminology alone. Presence is not demonstration; proximity is not "
        "integration; terminology is not application. Use low confidence and recommend "
        "human review when text, arrows, localization, or interpretation is uncertain. "
        "Low confidence does not change the best evidence-based score.\n"
    )


def compact_evidence_contract() -> str:
    """Describe repeated evidence fields once to keep provider prompts bounded."""
    extended = extend_grading_schema({})
    contract = {
        "apply_to_every_criterion": evidence_schema_fragment(),
        "apply_to_every_domain": {
            "visual_summary": {
                "strongest_visible_evidence": ["<concise visible evidence>"],
                "most_important_missing_connection": "<string or empty string>",
                "domain_confidence": "<number 0..1>",
                "human_review_recommended": "<boolean>",
            }
        },
        "top_level_multimodal_feedback": extended["multimodal_feedback"],
        "top_level_learning_feedback": extended["learning_feedback"],
    }
    return "\nMULTIMODAL JSON CONTRACT (apply these fields at the stated locations)\n" + _json_text(contract)


def evidence_only_recovery_prompt(
    original_result: Mapping[str, Any],
    output_template: Mapping[str, Any],
) -> str:
    return (
        "The rubric grading below is final and immutable. Inspect the same concept-map "
        "image and return ONLY the missing grounded multimodal fields. Do not change or "
        "restate scores, criterion explanations, domain decisions, or the overall decision. "
        "Do not invent evidence or bounding boxes. Use null bbox when localization is "
        "uncertain.\n\nORIGINAL GRADING\n"
        + _compact_grading_identity(original_result)
        + "\n\nREQUIRED EVIDENCE STRUCTURE\n"
        + _json_text(output_template)
        + compact_grounding_instructions()
        + "\nReturn one raw valid JSON object only."
    )


def evidence_recovery_template() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain, criteria in CRITERION_FIELDS.items():
        result[domain] = {
            criterion: copy.deepcopy(evidence_schema_fragment())
            for criterion in criteria
        }
        result[domain]["visual_summary"] = {
            "strongest_visible_evidence": ["<concise visible evidence>"],
            "most_important_missing_connection": "<concise missing connection or empty string>",
            "domain_confidence": "<number 0..1>",
            "human_review_recommended": "<boolean>",
        }
    result["multimodal_feedback"] = extend_grading_schema({})["multimodal_feedback"]
    result["learning_feedback"] = extend_grading_schema({})["learning_feedback"]
    return result


def _compact_grading_identity(result: Mapping[str, Any]) -> str:
    compact: dict[str, Any] = {
        "overall_meets_expectations": result.get("overall_meets_expectations")
    }
    for domain, criteria in CRITERION_FIELDS.items():
        section = result.get(domain)
        if not isinstance(section, Mapping):
            continue
        compact[domain] = {
            criterion: {
                "score": section.get(criterion, {}).get("score")
                if isinstance(section.get(criterion), Mapping)
                else None,
                "explanation": section.get(criterion, {}).get("explanation")
                if isinstance(section.get(criterion), Mapping)
                else None,
            }
            for criterion in criteria
        }
        compact[domain]["overall_decision"] = section.get("overall_decision")
        compact[domain]["if_no_explanation"] = section.get("if_no_explanation")
    return _json_text(compact)


def _json_text(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def valid_bbox(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, list) or len(value) != 4 or not all(_number(item) for item in value):
        return False
    x_min, y_min, x_max, y_max = (float(item) for item in value)
    return (
        all(0.0 <= item <= 1.0 for item in (x_min, y_min, x_max, y_max))
        and x_min < x_max
        and y_min < y_max
    )


def _confidence(value: Any) -> bool:
    return _number(value) and 0.0 <= float(value) <= 1.0


def normalize_multimodal_numbers(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert unambiguous numeric strings in visual fields; never infer values."""
    changes: list[dict[str, Any]] = []

    def normalize_number(container: dict[str, Any], field: str, path: str) -> None:
        value = container.get(field)
        if not isinstance(value, str):
            return
        stripped = value.strip()
        try:
            normalized = float(stripped)
        except ValueError:
            return
        container[field] = normalized
        changes.append({"field": path, "original": value, "normalized": normalized})

    def normalize_bbox(container: dict[str, Any], path: str) -> None:
        bbox = container.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return
        converted: list[Any] = []
        changed = False
        for value in bbox:
            if isinstance(value, str):
                try:
                    converted.append(float(value.strip()))
                    changed = True
                except ValueError:
                    return
            else:
                converted.append(value)
        if changed:
            container["bbox"] = converted
            changes.append({"field": path, "original": bbox, "normalized": converted})

    if not isinstance(result, dict):
        return changes
    for domain, criteria in CRITERION_FIELDS.items():
        section = result.get(domain)
        if not isinstance(section, dict):
            continue
        for criterion in criteria:
            item = section.get(criterion)
            if not isinstance(item, dict):
                continue
            path = f"{domain}.{criterion}"
            normalize_number(item, "criterion_confidence", f"{path}.criterion_confidence")
            for index, evidence in enumerate(item.get("supporting_evidence", [])):
                if not isinstance(evidence, dict):
                    continue
                normalize_number(
                    evidence,
                    "confidence",
                    f"{path}.supporting_evidence[{index}].confidence",
                )
                normalize_bbox(evidence, f"{path}.supporting_evidence[{index}].bbox")
        visual_summary = section.get("visual_summary")
        if isinstance(visual_summary, dict):
            normalize_number(
                visual_summary,
                "domain_confidence",
                f"{domain}.visual_summary.domain_confidence",
            )
    feedback = result.get("multimodal_feedback")
    if isinstance(feedback, dict):
        normalize_number(
            feedback,
            "overall_visual_confidence",
            "multimodal_feedback.overall_visual_confidence",
        )
        for field in ("strongest_regions", "highest_priority_improvements"):
            for index, item in enumerate(feedback.get(field, [])):
                if not isinstance(item, dict):
                    continue
                normalize_number(
                    item,
                    "confidence",
                    f"multimodal_feedback.{field}[{index}].confidence",
                )
                normalize_bbox(item, f"multimodal_feedback.{field}[{index}].bbox")
    for index, item in enumerate(result.get("learning_feedback", [])):
        if not isinstance(item, dict):
            continue
        normalize_number(item, "confidence", f"learning_feedback[{index}].confidence")
        normalize_bbox(item, f"learning_feedback[{index}].bbox")
    return changes


def validate_multimodal_feedback(result: Mapping[str, Any]) -> MultimodalValidation:
    warnings: list[str] = []
    missing: list[str] = []
    invalid_bbox_count = 0
    any_multimodal = False

    def require(condition: bool, path: str, message: str) -> None:
        if not condition:
            warnings.append(f"{path}: {message}")

    def check_bbox(value: Any, path: str) -> None:
        nonlocal invalid_bbox_count
        if not valid_bbox(value):
            invalid_bbox_count += 1
            warnings.append(f"{path}: invalid normalized bbox")

    for domain, criteria in CRITERION_FIELDS.items():
        section = result.get(domain)
        if not isinstance(section, Mapping):
            continue
        for criterion in criteria:
            item = section.get(criterion)
            if not isinstance(item, Mapping):
                continue
            path = f"{domain}.{criterion}"
            for field in CRITERION_EVIDENCE_FIELDS:
                if field not in item:
                    missing.append(f"{path}.{field}")
                else:
                    any_multimodal = True
            supporting = item.get("supporting_evidence")
            require(isinstance(supporting, list), f"{path}.supporting_evidence", "must be an array")
            if isinstance(supporting, list):
                require(len(supporting) <= 3, f"{path}.supporting_evidence", "maximum is 3")
                for index, evidence in enumerate(supporting):
                    evidence_path = f"{path}.supporting_evidence[{index}]"
                    require(isinstance(evidence, Mapping), evidence_path, "must be an object")
                    if not isinstance(evidence, Mapping):
                        continue
                    require(bool(str(evidence.get("evidence_text", "")).strip()), evidence_path, "evidence_text is required")
                    require(bool(str(evidence.get("location_description", "")).strip()), evidence_path, "location_description is required")
                    require(bool(str(evidence.get("relationship_type", "")).strip()), evidence_path, "relationship_type is required")
                    require(_confidence(evidence.get("confidence")), evidence_path, "confidence must be 0..1")
                    check_bbox(evidence.get("bbox"), f"{evidence_path}.bbox")
            missing_items = item.get("missing_evidence")
            require(isinstance(missing_items, list), f"{path}.missing_evidence", "must be an array")
            if isinstance(missing_items, list):
                for index, missing_item in enumerate(missing_items):
                    missing_path = f"{path}.missing_evidence[{index}]"
                    require(isinstance(missing_item, Mapping), missing_path, "must be an object")
                    if not isinstance(missing_item, Mapping):
                        continue
                    require(bool(str(missing_item.get("missing_relationship", "")).strip()), missing_path, "missing_relationship is required")
                    require(bool(str(missing_item.get("suggested_connection", "")).strip()), missing_path, "suggested_connection is required")
                    require(missing_item.get("importance") in IMPORTANCE_VALUES, missing_path, "invalid importance")
            require(_confidence(item.get("criterion_confidence")), f"{path}.criterion_confidence", "must be 0..1")
            require(isinstance(item.get("human_review_recommended"), bool), f"{path}.human_review_recommended", "must be boolean")
            if _confidence(item.get("criterion_confidence")) and float(item["criterion_confidence"]) < 0.60:
                require(
                    item.get("human_review_recommended") is True,
                    f"{path}.human_review_recommended",
                    "must be true when criterion_confidence is below 0.60",
                )

        visual_summary = section.get("visual_summary")
        if visual_summary is None:
            missing.append(f"{domain}.visual_summary")
        else:
            any_multimodal = True
            require(isinstance(visual_summary, Mapping), f"{domain}.visual_summary", "must be an object")
            if isinstance(visual_summary, Mapping):
                require(isinstance(visual_summary.get("strongest_visible_evidence"), list), f"{domain}.visual_summary", "strongest_visible_evidence must be an array")
                require(isinstance(visual_summary.get("most_important_missing_connection"), str), f"{domain}.visual_summary", "most_important_missing_connection must be a string")
                require(_confidence(visual_summary.get("domain_confidence")), f"{domain}.visual_summary", "domain_confidence must be 0..1")
                require(isinstance(visual_summary.get("human_review_recommended"), bool), f"{domain}.visual_summary", "human_review_recommended must be boolean")

    overall = result.get("multimodal_feedback")
    if overall is None:
        missing.append("multimodal_feedback")
    else:
        any_multimodal = True
        require(isinstance(overall, Mapping), "multimodal_feedback", "must be an object")
        if isinstance(overall, Mapping):
            strongest = overall.get("strongest_regions")
            improvements = overall.get("highest_priority_improvements")
            require(isinstance(strongest, list), "multimodal_feedback.strongest_regions", "must be an array")
            require(isinstance(improvements, list), "multimodal_feedback.highest_priority_improvements", "must be an array")
            if isinstance(strongest, list):
                require(len(strongest) <= 3, "multimodal_feedback.strongest_regions", "maximum is 3")
                for index, region in enumerate(strongest):
                    path = f"multimodal_feedback.strongest_regions[{index}]"
                    require(isinstance(region, Mapping), path, "must be an object")
                    if isinstance(region, Mapping):
                        require(bool(str(region.get("description", "")).strip()), path, "description is required")
                        require(_confidence(region.get("confidence")), path, "confidence must be 0..1")
                        check_bbox(region.get("bbox"), f"{path}.bbox")
            if isinstance(improvements, list):
                require(len(improvements) <= 3, "multimodal_feedback.highest_priority_improvements", "maximum is 3")
                for index, improvement in enumerate(improvements):
                    path = f"multimodal_feedback.highest_priority_improvements[{index}]"
                    require(isinstance(improvement, Mapping), path, "must be an object")
                    if isinstance(improvement, Mapping):
                        for field in ("current_state", "missing_bridge", "suggested_revision"):
                            require(bool(str(improvement.get(field, "")).strip()), path, f"{field} is required")
                        require(improvement.get("importance") in IMPORTANCE_VALUES, path, "invalid importance")
                        check_bbox(improvement.get("bbox"), f"{path}.bbox")
            require(_confidence(overall.get("overall_visual_confidence")), "multimodal_feedback", "overall_visual_confidence must be 0..1")
            require(isinstance(overall.get("human_review_recommended"), bool), "multimodal_feedback", "human_review_recommended must be boolean")

    learning = result.get("learning_feedback")
    if learning is None:
        missing.append("learning_feedback")
    else:
        any_multimodal = True
        require(isinstance(learning, list), "learning_feedback", "must be an array")
        if isinstance(learning, list):
            for index, item in enumerate(learning):
                path = f"learning_feedback[{index}]"
                require(isinstance(item, Mapping), path, "must be an object")
                if not isinstance(item, Mapping):
                    continue
                criterion = str(item.get("criterion", ""))
                require(any(criterion in fields for fields in CRITERION_FIELDS.values()), path, "unknown criterion")
                for field in ("observed_evidence", "guiding_question", "hint"):
                    require(bool(str(item.get(field, "")).strip()), path, f"{field} is required")
                require(_confidence(item.get("confidence")), path, "confidence must be 0..1")
                check_bbox(item.get("bbox"), f"{path}.bbox")

    complete = not missing and not warnings
    return MultimodalValidation(
        available=any_multimodal and complete,
        complete=complete,
        warnings=tuple(warnings),
        missing_fields=tuple(missing),
        invalid_bbox_count=invalid_bbox_count,
    )


def merge_recovered_evidence(
    original: Mapping[str, Any],
    recovered: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Copy only evidence fields from recovery; grading fields remain immutable."""
    merged = copy.deepcopy(dict(original))
    ignored: list[str] = []
    for domain, criteria in CRITERION_FIELDS.items():
        recovered_section = recovered.get(domain)
        target_section = merged.get(domain)
        if not isinstance(recovered_section, Mapping) or not isinstance(target_section, dict):
            continue
        for criterion in criteria:
            recovered_item = recovered_section.get(criterion)
            target_item = target_section.get(criterion)
            if not isinstance(recovered_item, Mapping) or not isinstance(target_item, dict):
                continue
            for field in CRITERION_EVIDENCE_FIELDS:
                if field in recovered_item:
                    target_item[field] = copy.deepcopy(recovered_item[field])
            for forbidden in ("score", "explanation"):
                if forbidden in recovered_item:
                    ignored.append(f"{domain}.{criterion}.{forbidden}")
        if "visual_summary" in recovered_section:
            target_section["visual_summary"] = copy.deepcopy(recovered_section["visual_summary"])
        for forbidden in ("overall_decision", "if_no_explanation"):
            if forbidden in recovered_section:
                ignored.append(f"{domain}.{forbidden}")
    for field in TOP_LEVEL_EVIDENCE_FIELDS:
        if field in recovered:
            merged[field] = copy.deepcopy(recovered[field])
    for forbidden in (
        "overall_meets_expectations",
        "strengths",
        "areas_for_improvement",
        "grading_notes",
        "map_file",
        "model",
    ):
        if forbidden in recovered:
            ignored.append(forbidden)
    return merged, tuple(ignored)


def multimodal_debug_metrics(result: Mapping[str, Any], validation: MultimodalValidation) -> dict[str, Any]:
    counts: dict[str, int] = {}
    confidences: dict[str, Any] = {}
    review_flags: dict[str, Any] = {}
    bbox_count = 0
    null_bbox_count = 0
    for domain, criteria in CRITERION_FIELDS.items():
        section = result.get(domain)
        if not isinstance(section, Mapping):
            continue
        for criterion in criteria:
            item = section.get(criterion)
            if not isinstance(item, Mapping):
                continue
            path = f"{domain}.{criterion}"
            evidence = item.get("supporting_evidence")
            counts[path] = len(evidence) if isinstance(evidence, list) else 0
            confidences[path] = item.get("criterion_confidence")
            review_flags[path] = item.get("human_review_recommended")
            if isinstance(evidence, list):
                for evidence_item in evidence:
                    if not isinstance(evidence_item, Mapping):
                        continue
                    if evidence_item.get("bbox") is None:
                        null_bbox_count += 1
                    elif valid_bbox(evidence_item.get("bbox")):
                        bbox_count += 1
    return {
        "multimodal_schema_version": MULTIMODAL_SCHEMA_VERSION,
        "supporting_evidence_count_by_criterion": counts,
        "bbox_count": bbox_count,
        "null_bbox_count": null_bbox_count,
        "invalid_bbox_count": validation.invalid_bbox_count,
        "criterion_confidence_values": confidences,
        "human_review_flags": review_flags,
        "multimodal_validation_warnings": list(validation.warnings),
        "multimodal_missing_fields": list(validation.missing_fields),
    }
