"""Stable Streamlit grading runner.

Active flow:
PDF upload -> first-page image -> model direct rubric grading -> JSON -> UI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from grading import grade_gemma, grade_llama


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web_demo"
DEBUG_DIR = OUTPUT_DIR / "debug"
EVALUATION_SUMMARY_DIR = PROJECT_ROOT / "outputs" / "evaluation_summary"
RAW_EVALUATION_DIR = EVALUATION_SUMMARY_DIR / "raw"
FAILURE_EVALUATION_DIR = EVALUATION_SUMMARY_DIR / "failures"

MODEL_MODULES = {
    "Gemma": grade_gemma,
    "Qwen 3.6 27B": grade_llama,
}

MODEL_IDS = {
    "Gemma": grade_gemma.MODEL,
    "Qwen 3.6 27B": grade_llama.MODEL,
}

CATEGORY_FIELDS = grade_gemma.CATEGORY_FIELDS
DOMAIN_LABELS = {
    "knowledge_acquisition": "Knowledge Acquisition",
    "integration": "Integration",
    "application": "Application",
    "transfer": "Transfer",
}

FORBIDDEN_DECISION_TEXT = (
    "Partial",
    "Partially Meets",
    "Borderline",
    "Maybe",
    "Almost",
    "Meets some expectations",
)


class GradingError(RuntimeError):
    """A user-facing failure while grading a concept map."""


class ModelResponseError(GradingError):
    """A model returned no usable grading response."""

    def __init__(self, message: str, raw_response: Any | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class MalformedResultError(GradingError):
    """A model response is not valid grading JSON."""


@dataclass(frozen=True)
class EvaluationResult:
    """One model's parsed, validated result and web-demo output location."""

    model_name: str
    model_id: str
    data: dict[str, Any]
    output_path: Path
    evaluated_at: str | None = None
    reference_materials_used: bool = False
    reference_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationFailure:
    """One model's failed result and persisted debug location."""

    model_name: str
    model_id: str
    error_message: str
    debug_path: Path


EvaluationOutcome = EvaluationResult | EvaluationFailure


def selected_model_names(selection: str) -> list[str]:
    """Translate the UI selection into active model names."""
    normalized = (
        selection.strip()
        .replace("‑", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    routes = {
        "Gemma": ["Gemma"],
        "Qwen 3.6 27B": ["Qwen 3.6 27B"],
        "Both": ["Gemma", "Qwen 3.6 27B"],
    }
    try:
        return routes[normalized]
    except KeyError as exc:
        raise GradingError(f"Unknown model selection: {normalized}") from exc


def model_debug_lines(model_names: Iterable[str] | None = None) -> list[str]:
    """Return internal provider/model debug lines; app.py does not render these."""
    names = list(model_names) if model_names is not None else ["Gemma", "Qwen 3.6 27B"]
    lines: list[str] = []
    for name in names:
        module = MODEL_MODULES.get(name)
        if module is None:
            continue
        lines.append(f"{name} provider: {module.PROVIDER}")
        lines.append(f"{name} model: {module.MODEL}")
    return lines


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned[:60] or "concept_map"


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", model_name.lower()).strip("_") or "model"


def _failure_suffix(model_name: str) -> str:
    if model_name == "Qwen 3.6 27B":
        return "qwen36_failure"
    return f"{_model_slug(model_name)}_failure"


def _result_suffix(model_name: str) -> str:
    """Keep saved result filenames stable and easy to identify."""
    if model_name == "Qwen 3.6 27B":
        return "qwen36"
    return _model_slug(model_name)


def _strip_json_fences(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$", "", text)


def _cleaned_json_text(raw_text: str) -> str:
    text = _strip_json_fences(raw_text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0).strip() if match else text


def _contains_forbidden_decision_text(value: Any) -> bool:
    if isinstance(value, str):
        return any(text.lower() in value.lower() for text in FORBIDDEN_DECISION_TEXT)
    if isinstance(value, dict):
        return any(_contains_forbidden_decision_text(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_decision_text(item) for item in value)
    return False


def _require_yes_no(value: Any, field_path: str) -> None:
    if value not in {"Yes", "No"}:
        raise MalformedResultError(f"'{field_path}' must be exactly 'Yes' or 'No'.")


def _normalize_decision_value(value: Any) -> tuple[str, str | None]:
    """Normalize model decision labels to the rubric's binary Yes/No values."""
    if isinstance(value, str):
        original = value.strip()
    elif value is None:
        original = ""
    else:
        original = str(value).strip()

    normalized = re.sub(r"\s+", " ", original).strip().lower()
    if normalized in {"yes", "meets", "meets expectations"}:
        return "Yes", None
    if normalized in {"no", "does not meet", "does not meet expectations"}:
        return "No", None
    return "No", original or repr(value)


def _append_grading_note(result: dict[str, Any], note: str) -> None:
    current = result.get("grading_notes", "")
    if isinstance(current, list):
        current_text = " ".join(str(item) for item in current if str(item).strip())
    else:
        current_text = str(current).strip()
    result["grading_notes"] = f"{current_text} {note}".strip() if current_text else note


def _criterion_label(field_name: str) -> str:
    return field_name.replace("_", " ").title()


def _short_reason_from_item(item: dict[str, Any]) -> str:
    explanation = str(item.get("explanation", "")).strip()
    if explanation:
        return explanation.rstrip(".")
    evidence = item.get("evidence_from_map")
    if isinstance(evidence, list):
        useful_evidence = [
            str(value).strip()
            for value in evidence
            if str(value).strip()
            and str(value).strip() != "No clear evidence found in the concept map."
        ]
        if useful_evidence:
            return useful_evidence[0].rstrip(".")
    elif isinstance(evidence, str) and evidence.strip():
        return evidence.strip().rstrip(".")
    return ""


def _join_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return " and ".join([", ".join(labels[:-1]), labels[-1]]) if len(labels) > 2 else " and ".join(labels)


def _generated_if_no_explanation(group: str, section: dict[str, Any]) -> str:
    fallback = (
        "This domain does not meet expectations based on the criterion scores "
        "and available evidence."
    )
    scored_items: list[tuple[int, str, dict[str, Any]]] = []
    for field in CATEGORY_FIELDS.get(group, []):
        item = section.get(field)
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if isinstance(score, int) and not isinstance(score, bool):
            scored_items.append((score, field, item))

    if not scored_items:
        return fallback

    scored_items.sort(key=lambda value: value[0])
    lowest_items = scored_items[:2]
    labels = [_criterion_label(field) for _, field, _ in lowest_items]
    label_text = _join_labels(labels)
    if not label_text:
        return fallback

    domain_label = DOMAIN_LABELS.get(group, _criterion_label(group))
    reasons = [
        _short_reason_from_item(item)
        for _, _, item in lowest_items
        if _short_reason_from_item(item)
    ]
    if reasons:
        reason_text = "; ".join(reasons[:2])
        verb = "was" if len(labels) == 1 else "were"
        return (
            f"{domain_label} does not meet expectations because {label_text} "
            f"{verb} incomplete: {reason_text}."
        )
    verb = "was" if len(labels) == 1 else "were"
    return f"{domain_label} does not meet expectations because {label_text} {verb} incomplete."


def _normalize_if_no_explanations(result: dict[str, Any]) -> None:
    """Fill missing domain explanations before schema validation."""
    for group in CATEGORY_FIELDS:
        section = result.get(group)
        if not isinstance(section, dict):
            continue
        if section.get("overall_decision") != "No":
            continue
        explanation = section.get("if_no_explanation")
        if isinstance(explanation, str) and explanation.strip():
            continue
        if explanation is not None and not isinstance(explanation, str):
            continue
        section["if_no_explanation"] = _generated_if_no_explanation(group, section)


def _normalize_decision_fields(result: dict[str, Any]) -> None:
    """Normalize binary decision labels before schema validation."""
    converted: list[str] = []

    field_paths: list[tuple[dict[str, Any], str, str]] = [
        (result, "overall_meets_expectations", "overall_meets_expectations")
    ]
    for group in CATEGORY_FIELDS:
        section = result.get(group)
        if isinstance(section, dict):
            field_paths.append((section, "overall_decision", f"{group}.overall_decision"))

    for container, field, field_path in field_paths:
        original_value = container.get(field)
        normalized_value, ambiguous_original = _normalize_decision_value(original_value)
        container[field] = normalized_value
        if ambiguous_original is not None:
            converted.append(f"{field_path}={ambiguous_original!r}")

    if converted:
        _append_grading_note(
            result,
            "Decision labels normalized to Yes/No; original ambiguous values: "
            + "; ".join(converted)
            + ".",
        )


def parse_model_json(raw_text: str, normalize_decisions: bool = False) -> dict[str, Any]:
    """Parse and validate the final grading JSON schema."""
    if not raw_text or not raw_text.strip():
        raise ModelResponseError("The model returned an empty response.")

    cleaned = _cleaned_json_text(raw_text)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise MalformedResultError(f"The model response was not valid JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise MalformedResultError("The model response JSON must be an object.")
    if normalize_decisions:
        _normalize_decision_fields(result)
    _normalize_if_no_explanations(result)
    if not normalize_decisions and _contains_forbidden_decision_text(result):
        raise MalformedResultError(
            "The model result contains a forbidden non-binary decision label."
        )

    required_top_level = [
        *CATEGORY_FIELDS.keys(),
        "overall_meets_expectations",
        "strengths",
        "areas_for_improvement",
        "grading_notes",
    ]
    missing = [field for field in required_top_level if field not in result]
    if missing:
        raise MalformedResultError(
            "The model result is missing required fields: " + ", ".join(missing)
        )

    _require_yes_no(result.get("overall_meets_expectations"), "overall_meets_expectations")

    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            raise MalformedResultError(f"'{group}' must be a JSON object.")
        _require_yes_no(section.get("overall_decision"), f"{group}.overall_decision")
        if section.get("overall_decision") == "No" and not str(
            section.get("if_no_explanation", "")
        ).strip():
            raise MalformedResultError(
                f"'{group}.if_no_explanation' is required when overall_decision is 'No'."
            )
        for field in fields:
            item = section.get(field)
            if not isinstance(item, dict):
                raise MalformedResultError(f"'{group}.{field}' must be a JSON object.")
            score = item.get("score")
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 4:
                raise MalformedResultError(
                    f"'{group}.{field}.score' must be an integer from 1 to 4."
                )
            if not isinstance(item.get("explanation"), str):
                raise MalformedResultError(
                    f"'{group}.{field}.explanation' must be a string."
                )
    return result


def _debug_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for method_name in ("model_dump_json", "json"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return method(indent=2) if method_name == "model_dump_json" else method()
            except TypeError:
                try:
                    return method()
                except Exception:
                    pass
            except Exception:
                pass
    return repr(value)


def _save_failed_response(
    *,
    timestamp: str,
    run_id: str,
    file_stem: str,
    model_name: str,
    model_id: str,
    error_message: str,
    raw_response: Any,
) -> Path:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    debug_path = (
        DEBUG_DIR
        / f"{timestamp}_{run_id}_{file_stem}_{_failure_suffix(model_name)}.json"
    )
    debug_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_id": model_id,
                "error": error_message,
                "raw_response": _debug_text(raw_response),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return debug_path


def _save_successful_evaluation(
    *,
    original_filename: str,
    evaluated_at: str,
    model_name: str,
    model_id: str,
    data: dict[str, Any],
    reference_materials_used: bool,
    reference_files: tuple[str, ...],
) -> Path:
    """Persist one validated model result, replacing only an older success."""
    RAW_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_EVALUATION_DIR / (
        f"{_safe_stem(original_filename)}_{_result_suffix(model_name)}.json"
    )
    payload = {
        "map_file": Path(original_filename).name,
        "evaluated_at": evaluated_at,
        "model_label": model_name,
        "model_id": model_id,
        "reference_materials_used": reference_materials_used,
        "reference_files": list(reference_files),
        "result": data,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_evaluation_results(
    results: Iterable[EvaluationOutcome], original_filename: str
) -> list[str]:
    """Manually save the successful results currently displayed in Streamlit.

    Failed outcomes are deliberately ignored, so they can never replace an
    already saved successful result for the same map and model.
    """
    saved_models: list[str] = []
    for result in results:
        if not isinstance(result, EvaluationResult):
            continue
        _save_successful_evaluation(
            original_filename=original_filename,
            evaluated_at=result.evaluated_at or datetime.now(timezone.utc).isoformat(),
            model_name=result.model_name,
            model_id=result.model_id,
            data=result.data,
            reference_materials_used=result.reference_materials_used,
            reference_files=result.reference_files,
        )
        saved_models.append(result.model_name)
    return saved_models


def _save_evaluation_failure(
    *,
    timestamp: str,
    run_id: str,
    original_filename: str,
    evaluated_at: str,
    model_name: str,
    model_id: str,
    error_message: str,
    debug_path: Path,
) -> Path:
    """Persist a failed run without touching a previously saved success."""
    FAILURE_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    failure_path = FAILURE_EVALUATION_DIR / (
        f"{timestamp}_{run_id}_{_safe_stem(original_filename)}_"
        f"{_failure_suffix(model_name)}.json"
    )
    failure_path.write_text(
        json.dumps(
            {
                "map_file": Path(original_filename).name,
                "evaluated_at": evaluated_at,
                "model_label": model_name,
                "model_id": model_id,
                "error": error_message,
                "debug_path": str(debug_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return failure_path


def run_evaluation(
    pdf_path: Path,
    model_names: Iterable[str],
    original_filename: str,
    progress_callback: Any | None = None,
    reference_materials: list[dict[str, str]] | None = None,
) -> list[EvaluationOutcome]:
    """Run one direct grading call for each selected model."""
    names = list(model_names)
    if not names:
        raise GradingError("Select at least one model.")
    unknown = [name for name in names if name not in MODEL_MODULES]
    if unknown:
        raise GradingError("Unknown model(s): " + ", ".join(unknown))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    evaluated_at_datetime = datetime.now(timezone.utc)
    evaluated_at = evaluated_at_datetime.isoformat()
    timestamp = evaluated_at_datetime.strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex[:8]
    file_stem = _safe_stem(original_filename)
    map_file = Path(original_filename).name
    reference_files = tuple(
        str(item.get("filename", "")).strip()
        for item in reference_materials or []
        if str(item.get("filename", "")).strip()
    )
    results: list[EvaluationOutcome] = []

    for model_name in names:
        module = MODEL_MODULES[model_name]
        model_id = module.MODEL
        debug_prefix = DEBUG_DIR / f"{timestamp}_{run_id}_{file_stem}_{_model_slug(model_name)}"
        raw_response: Any = None
        try:
            if progress_callback:
                progress_callback(f"Running {model_name} grading")
            grade = module.grade_pdf(
                pdf_path,
                map_file,
                debug_prefix,
                reference_materials=reference_materials,
            )
            raw_response = grade.get("response")
            data = parse_model_json(
                str(grade["cleaned_text"]),
                normalize_decisions=True,
            )

            output_path = (
                OUTPUT_DIR
                / f"{timestamp}_{run_id}_{file_stem}_{_model_slug(model_name)}.json"
            )
            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            debug_path = Path(f"{debug_prefix}_debug.json")
            debug_payload = {
                **grade.get("debug", {}),
                "prompt_path": str(grade.get("prompt_path")),
                "raw_path": str(grade.get("raw_path")),
                "output_path": str(output_path),
            }
            debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

            results.append(
                EvaluationResult(
                    model_name,
                    model_id,
                    data,
                    output_path,
                    evaluated_at,
                    bool(reference_files),
                    reference_files,
                )
            )
        except Exception as exc:
            raw = getattr(exc, "raw_response", None)
            if raw is None:
                raw = raw_response
            debug_path = _save_failed_response(
                timestamp=timestamp,
                run_id=run_id,
                file_stem=file_stem,
                model_name=model_name,
                model_id=model_id,
                error_message=str(exc),
                raw_response=raw,
            )
            _save_evaluation_failure(
                timestamp=timestamp,
                run_id=run_id,
                original_filename=original_filename,
                evaluated_at=evaluated_at,
                model_name=model_name,
                model_id=model_id,
                error_message=str(exc),
                debug_path=debug_path,
            )
            results.append(
                EvaluationFailure(
                    model_name=model_name,
                    model_id=model_id,
                    error_message=str(exc),
                    debug_path=debug_path,
                )
            )
    return results
