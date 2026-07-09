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

MODEL_MODULES = {
    "Gemma": grade_gemma,
    "Llama": grade_llama,
}

MODEL_IDS = {
    "Gemma": grade_gemma.MODEL,
    "Llama": grade_llama.MODEL,
}

CATEGORY_FIELDS = grade_gemma.CATEGORY_FIELDS

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
    """One model's parsed result and persisted output location."""

    model_name: str
    model_id: str
    data: dict[str, Any]
    output_path: Path


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
        "Llama": ["Llama"],
        "Both": ["Gemma", "Llama"],
    }
    try:
        return routes[normalized]
    except KeyError as exc:
        raise GradingError(f"Unknown model selection: {normalized}") from exc


def model_debug_lines(model_names: Iterable[str] | None = None) -> list[str]:
    """Return internal provider/model debug lines; app.py does not render these."""
    names = list(model_names) if model_names is not None else ["Gemma", "Llama"]
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


def parse_model_json(raw_text: str) -> dict[str, Any]:
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
    if _contains_forbidden_decision_text(result):
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
            evidence = item.get("evidence_from_map")
            if not isinstance(evidence, list):
                raise MalformedResultError(
                    f"'{group}.{field}.evidence_from_map' must be a list."
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
        / f"{timestamp}_{run_id}_{file_stem}_{_model_slug(model_name)}_failure.json"
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


def run_evaluation(
    pdf_path: Path,
    model_names: Iterable[str],
    original_filename: str,
    progress_callback: Any | None = None,
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex[:8]
    file_stem = _safe_stem(original_filename)
    map_file = Path(original_filename).name
    results: list[EvaluationOutcome] = []

    for model_name in names:
        module = MODEL_MODULES[model_name]
        model_id = module.MODEL
        debug_prefix = DEBUG_DIR / f"{timestamp}_{run_id}_{file_stem}_{_model_slug(model_name)}"
        raw_response: Any = None
        try:
            if progress_callback:
                progress_callback(f"Running {model_name} grading")
            grade = module.grade_pdf(pdf_path, map_file, debug_prefix)
            raw_response = grade.get("response")
            data = parse_model_json(str(grade["cleaned_text"]))

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

            results.append(EvaluationResult(model_name, model_id, data, output_path))
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
            results.append(
                EvaluationFailure(
                    model_name=model_name,
                    model_id=model_id,
                    error_message=str(exc),
                    debug_path=debug_path,
                )
            )

    return results

