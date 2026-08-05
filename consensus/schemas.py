"""Validation for review and consensus envelopes."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from consensus.comparison import (
    COMPARISON_FIELD_BY_PATH,
    get_path,
    set_path,
)
from interface.grading_runner import parse_model_json


ALLOWED_REVIEW_ACTIONS = {
    "revised",
    "defended",
    "agreed_with_peer",
    "unresolved_uncertainty",
}
ALLOWED_CONSENSUS_STATUSES = {
    "complete",
    "complete_with_human_review",
    "unavailable",
}


class ConsensusValidationError(RuntimeError):
    pass


class ConsensusResolutionConsistencyError(ConsensusValidationError):
    """Resolution values disagree with the authoritative model grading."""

    def __init__(self, mismatches: list[dict[str, Any]]) -> None:
        self.mismatches = copy.deepcopy(mismatches)
        super().__init__(
            "Consensus resolution values do not match consensus_grading: "
            + json.dumps({"mismatches": mismatches}, separators=(",", ":"))
        )


@dataclass(frozen=True)
class ValidatedCrossReview:
    payload: dict[str, Any]
    reviewed_grading: dict[str, Any]
    warnings: tuple[str, ...]
    changed_field_paths: tuple[str, ...]


def clean_json_object(raw_text: str) -> dict[str, Any]:
    if not raw_text or not raw_text.strip():
        raise ConsensusValidationError("The model returned empty content.")
    text = re.sub(r"^\s*```(?:json)?\s*", "", raw_text.strip(), flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConsensusValidationError(f"The model response was not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConsensusValidationError("The model response must be a JSON object.")
    return value


def validate_complete_grading(
    value: Any,
    *,
    object_name: str = "reviewed_grading",
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConsensusValidationError(f"{object_name} must be a JSON object.")
    try:
        grading = parse_model_json(
            json.dumps(value, separators=(",", ":")),
            normalize_decisions=False,
        )
    except Exception as exc:
        readable_name = object_name.replace("_", " ")
        raise ConsensusValidationError(f"Incomplete {readable_name}: {exc}") from exc
    return grading


def _preserve_undisputed_grading_fields(
    reviewed: dict[str, Any],
    initial_own: Mapping[str, Any],
    disputed_paths: set[str],
) -> tuple[set[str], list[str]]:
    """Reject model edits outside disputed grading fields without authoring grades."""
    rejected: set[str] = set()
    warnings: list[str] = []

    def preserve(path: str) -> None:
        if get_path(reviewed, path) == get_path(initial_own, path):
            return
        set_path(reviewed, path, copy.deepcopy(get_path(initial_own, path)))
        rejected.add(path)
        warnings.append(f"Unauthorized undisputed field change rejected: {path}")

    preserve("map_file")
    preserve("model")
    for path, field in COMPARISON_FIELD_BY_PATH.items():
        if path not in disputed_paths:
            preserve(path)
        if field.disagreement_type == "criterion_score":
            explanation_path = path.removesuffix(".score") + ".explanation"
            if path not in disputed_paths:
                preserve(explanation_path)
        elif field.disagreement_type == "domain_decision":
            if path not in disputed_paths:
                preserve(path.removesuffix(".overall_decision") + ".if_no_explanation")
    for path in ("strengths", "areas_for_improvement", "grading_notes"):
        preserve(path)
    return rejected, warnings


def validate_cross_review(
    payload: Mapping[str, Any],
    *,
    reviewer_model: str,
    reviewed_peer_model: str,
    initial_own: Mapping[str, Any],
    initial_peer: Mapping[str, Any],
    initial_disagreement_paths: set[str],
) -> ValidatedCrossReview:
    value = copy.deepcopy(dict(payload))
    if value.get("reviewer_model") != reviewer_model:
        raise ConsensusValidationError("reviewer_model does not match the requested reviewer.")
    if value.get("reviewed_peer_model") != reviewed_peer_model:
        raise ConsensusValidationError("reviewed_peer_model does not match the peer.")
    reviewed = validate_complete_grading(value.get("reviewed_grading"))
    warnings: list[str] = []

    field_reviews = value.get("field_reviews")
    if not isinstance(field_reviews, list):
        raise ConsensusValidationError("field_reviews must be an array.")
    reviews_by_path: dict[str, Mapping[str, Any]] = {}
    for item in field_reviews:
        if not isinstance(item, Mapping):
            raise ConsensusValidationError("Every field review must be an object.")
        path = str(item.get("path", ""))
        if path not in COMPARISON_FIELD_BY_PATH:
            raise ConsensusValidationError(f"Unknown reviewed field path: {path}")
        if path in reviews_by_path:
            raise ConsensusValidationError(f"Duplicate field review path: {path}")
        if item.get("action") not in ALLOWED_REVIEW_ACTIONS:
            raise ConsensusValidationError(f"Invalid review action for {path}.")
        if not str(item.get("reason", "")).strip():
            raise ConsensusValidationError(f"A criterion-specific review reason is required for {path}.")
        if item.get("initial_own_value") != get_path(initial_own, path):
            raise ConsensusValidationError(f"initial_own_value does not match {path}.")
        if item.get("peer_value") != get_path(initial_peer, path):
            raise ConsensusValidationError(f"peer_value does not match {path}.")
        reviews_by_path[path] = item
    missing_reviews = initial_disagreement_paths - set(reviews_by_path)
    if missing_reviews:
        raise ConsensusValidationError(
            "Every disputed field requires field_reviews metadata: "
            + ", ".join(sorted(missing_reviews))
        )

    rejected_unauthorized_paths, preservation_warnings = (
        _preserve_undisputed_grading_fields(
            reviewed,
            initial_own,
            initial_disagreement_paths,
        )
    )
    warnings.extend(preservation_warnings)

    changed_paths = tuple(
        path
        for path in COMPARISON_FIELD_BY_PATH
        if get_path(reviewed, path) != get_path(initial_own, path)
    )
    for path in changed_paths:
        review = reviews_by_path.get(path)
        if review is None:
            raise ConsensusValidationError(f"Changed field lacks field_reviews metadata: {path}")
        if review.get("reviewed_own_value") != get_path(reviewed, path):
            raise ConsensusValidationError(f"reviewed_own_value does not match {path}.")
        if review.get("action") not in {"revised", "agreed_with_peer"}:
            raise ConsensusValidationError(f"Changed field has inconsistent action: {path}")
        if (
            review.get("action") == "agreed_with_peer"
            and get_path(reviewed, path) != get_path(initial_peer, path)
        ):
            raise ConsensusValidationError(
                f"agreed_with_peer does not match the peer value: {path}"
            )

    declared_changed = value.get("changed_field_paths")
    if not isinstance(declared_changed, list):
        raise ConsensusValidationError("changed_field_paths must be an array.")
    declared_set = set(declared_changed)
    if not set(changed_paths).issubset(declared_set):
        raise ConsensusValidationError("changed_field_paths does not match the reviewed grading.")
    unexpected_declared = declared_set - set(changed_paths)
    if unexpected_declared - rejected_unauthorized_paths:
        raise ConsensusValidationError("changed_field_paths does not match the reviewed grading.")
    value["changed_field_paths"] = list(changed_paths)
    declared_defended = value.get("defended_field_paths")
    if not isinstance(declared_defended, list):
        raise ConsensusValidationError("defended_field_paths must be an array.")
    for path in declared_defended:
        review = reviews_by_path.get(path)
        if review is None or review.get("action") != "defended":
            raise ConsensusValidationError(f"defended_field_paths is inconsistent for {path}.")
        if get_path(reviewed, path) != get_path(initial_own, path):
            raise ConsensusValidationError(f"A defended field changed value: {path}.")
    confidence = value.get("review_confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= float(confidence) <= 1
    ):
        raise ConsensusValidationError("review_confidence must be between 0 and 1.")

    value["reviewed_grading"] = reviewed
    return ValidatedCrossReview(value, reviewed, tuple(warnings), changed_paths)


def validate_consensus(
    payload: Mapping[str, Any],
    *,
    initial_disagreement_paths: set[str],
    required_resolution_paths: set[str] | None = None,
    expected_map_file: str | None = None,
    expected_model: str | None = None,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    status = value.get("consensus_status")
    if status not in ALLOWED_CONSENSUS_STATUSES:
        raise ConsensusValidationError("Invalid consensus_status.")
    if status == "unavailable":
        raise ConsensusValidationError(
            "A model-authored consensus response cannot use status unavailable or omit consensus_grading."
        )
    grading = validate_complete_grading(
        value.get("consensus_grading"),
        object_name="consensus_grading",
    )
    if expected_map_file is not None and grading.get("map_file") != expected_map_file:
        raise ConsensusValidationError(
            "consensus_grading.map_file does not match the evaluated map context."
        )
    if expected_model is not None and grading.get("model") != expected_model:
        raise ConsensusValidationError(
            "consensus_grading.model does not match the configured consensus model label."
        )
    resolutions = value.get("criterion_resolutions")
    unresolved = value.get("unresolved_disagreements")
    if not isinstance(resolutions, list) or not isinstance(unresolved, list):
        raise ConsensusValidationError(
            "criterion_resolutions and unresolved_disagreements must be arrays."
        )
    required_paths = (
        set(initial_disagreement_paths)
        if required_resolution_paths is None
        else set(required_resolution_paths)
    )
    seen_paths: set[str] = set()
    duplicate_paths: set[str] = set()
    extra_paths: set[str] = set()
    for item in resolutions:
        if not isinstance(item, Mapping):
            raise ConsensusValidationError("Every criterion resolution must be an object.")
        path = str(item.get("path", ""))
        if path not in required_paths:
            extra_paths.add(path)
            continue
        if path in seen_paths:
            duplicate_paths.add(path)
            continue
        seen_paths.add(path)
    if duplicate_paths:
        raise ConsensusValidationError(
            "Duplicate resolution paths: " + ", ".join(sorted(duplicate_paths))
        )
    if extra_paths:
        raise ConsensusValidationError(
            "Unknown or extra resolution paths: " + ", ".join(sorted(extra_paths))
        )
    missing_paths = required_paths - seen_paths
    if missing_paths:
        raise ConsensusValidationError(
            "Missing required resolution paths: " + ", ".join(sorted(missing_paths))
        )

    mismatches: list[dict[str, Any]] = []
    for item in resolutions:
        path = str(item.get("path", ""))
        if item.get("status") not in {"resolved", "unresolved"}:
            raise ConsensusValidationError(f"Invalid resolution status for {path}.")
        grading_value = get_path(grading, path)
        resolution_value = item.get("consensus_value")
        if (
            type(resolution_value) is not type(grading_value)
            or resolution_value != grading_value
        ):
            mismatches.append(
                {
                    "path": path,
                    "grading_value": grading_value,
                    "resolution_value": resolution_value,
                    "grading_type": type(grading_value).__name__,
                    "resolution_type": type(resolution_value).__name__,
                }
            )
        confidence = item.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= float(confidence) <= 1
        ):
            raise ConsensusValidationError(f"Invalid resolution confidence for {path}.")
        if not str(item.get("resolution_basis", "")).strip():
            raise ConsensusValidationError(f"resolution_basis is required for {path}.")
    if mismatches:
        raise ConsensusResolutionConsistencyError(mismatches)
    unresolved_paths: set[str] = set()
    for item in unresolved:
        if not isinstance(item, Mapping):
            raise ConsensusValidationError("Every unresolved disagreement must be an object.")
        path = str(item.get("path", ""))
        if path not in initial_disagreement_paths:
            raise ConsensusValidationError(f"Unresolved path was not initially disputed: {path}")
        if item.get("status") != "unresolved":
            raise ConsensusValidationError(f"Unresolved item has invalid status: {path}")
        if item.get("human_review_recommended") is not True:
            raise ConsensusValidationError(
                f"Human review is required for unresolved field: {path}"
            )
        if not str(item.get("reason", "")).strip():
            raise ConsensusValidationError(f"Unresolved reason is required for {path}.")
        unresolved_paths.add(path)
    for item in resolutions:
        path = str(item["path"])
        should_be_unresolved = item.get("status") == "unresolved"
        if should_be_unresolved != (path in unresolved_paths):
            raise ConsensusValidationError(
                f"Resolution and unresolved metadata disagree for {path}."
            )
    confidence = value.get("consensus_confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= float(confidence) <= 1
    ):
        raise ConsensusValidationError("consensus_confidence must be between 0 and 1.")
    if unresolved:
        if status != "complete_with_human_review" or value.get("human_review_recommended") is not True:
            raise ConsensusValidationError(
                "Unresolved disagreements require complete_with_human_review and human_review_recommended=true."
            )
    elif status != "complete":
        raise ConsensusValidationError("A fully resolved consensus must use status complete.")
    if not isinstance(value.get("human_review_recommended"), bool):
        raise ConsensusValidationError("human_review_recommended must be boolean.")
    if not isinstance(value.get("consensus_notes"), str):
        raise ConsensusValidationError("consensus_notes must be a string.")
    value["consensus_grading"] = grading
    return value
