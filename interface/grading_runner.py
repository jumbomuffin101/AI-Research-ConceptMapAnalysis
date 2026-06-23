"""Upload-driven grading pipeline used by the Streamlit demo.

The command-line graders remain independent. This module mirrors their proven
PDF rendering and OpenRouter request flow while accepting any uploaded PDF.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import fitz
from dotenv import load_dotenv

from grading import grade_nemotron, grade_qwen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web_demo"
DEBUG_DIR = OUTPUT_DIR / "debug"

GRADER_MODULES = {
    "Qwen": grade_qwen,
    "Nemotron": grade_nemotron,
}

MODEL_SELECTION_ALIASES = {
    "Qwen (Recommended)": "Qwen",
    "Nemotron (Experimental)": "Nemotron",
}

MODEL_CONFIGS = {
    "Qwen": {
        "model_id": grade_qwen.MODEL,
    },
    "Nemotron": {
        "model_id": grade_nemotron.MODEL,
    },
}

CATEGORY_FIELDS = {
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


class GradingError(RuntimeError):
    """A user-facing failure while grading a concept map."""


class InvalidPDFError(GradingError):
    """The uploaded file cannot be read as a PDF."""


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
    """Translate the UI selection into model registry keys."""
    if selection == "Both":
        return ["Qwen", "Nemotron"]
    selection = MODEL_SELECTION_ALIASES.get(selection, selection)
    if selection not in MODEL_CONFIGS:
        raise GradingError(f"Unknown model selection: {selection}")
    return [selection]


def render_pdf_image(pdf_path: Path, model_name: str) -> str:
    """Render the uploaded PDF using the selected grader's existing helper."""
    grader = GRADER_MODULES[model_name]
    try:
        return grader.pdf_to_base64(pdf_path)
    except (
        fitz.FileDataError,
        fitz.EmptyFileError,
        RuntimeError,
        ValueError,
        IndexError,
    ) as exc:
        raise InvalidPDFError("The uploaded file is not a valid, readable PDF.") from exc


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Extract, parse, and minimally validate grading JSON from a model response."""
    if not raw_text or not raw_text.strip():
        raise ModelResponseError("The model returned an empty response.")

    text = re.sub(r"^\s*```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        raise MalformedResultError("The model response did not contain a JSON object.")

    try:
        result, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise MalformedResultError(
            f"The model returned malformed JSON ({exc.msg}, line {exc.lineno})."
        ) from exc

    if not isinstance(result, dict):
        raise MalformedResultError("The model result must be a JSON object.")

    missing = [
        key
        for key in (*CATEGORY_FIELDS.keys(), "overall_map_meets_expectations")
        if key not in result
    ]
    if missing:
        raise MalformedResultError(
            "The model result is missing required fields: " + ", ".join(missing)
        )

    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            raise MalformedResultError(f"'{group}' must be a JSON object.")
        for field in fields:
            item = section.get(field)
            score = item.get("score") if isinstance(item, dict) else None
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 4:
                raise MalformedResultError(
                    f"'{group}.{field}.score' must be an integer from 1 to 4."
                )
    return result


def _response_to_debug_text(response: Any) -> str:
    """Convert an SDK response object into a debug-safe text payload."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    for method_name in ("model_dump_json", "json"):
        method = getattr(response, method_name, None)
        if callable(method):
            try:
                return method(indent=2)
            except TypeError:
                try:
                    return method()
                except Exception:
                    pass
            except Exception:
                pass
    return str(response)


def _save_failed_response(
    *,
    timestamp: str,
    run_id: str,
    file_stem: str,
    model_name: str,
    model_id: str,
    error_message: str,
    raw_response: Any | None,
) -> Path:
    """Persist failed model content for later debugging."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    debug_path = DEBUG_DIR / (
        f"{timestamp}_{run_id}_{file_stem}_{model_name.lower()}_failure.json"
    )
    payload = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_id": model_id,
        "error_message": error_message,
        "raw_response": _response_to_debug_text(raw_response),
    }
    debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return debug_path


def _ensure_api_key() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise GradingError(
            "OPENROUTER_API_KEY is missing. Add it to the environment or project .env file."
        )


def _request_model(
    model_name: str,
    prompt: str,
    image: str,
) -> tuple[str, str]:
    grader = GRADER_MODULES[model_name]

    try:
        client = grader.create_client()
        response = grader.request_grade(client, prompt, image)
    except Exception as exc:
        raise ModelResponseError(
            f"{model_name} API request failed: {exc}",
            raw_response=repr(exc),
        ) from exc

    if not response.choices:
        raise ModelResponseError(
            f"{model_name} returned no response choices.",
            raw_response=response,
        )
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ModelResponseError(
            f"{model_name} returned a malformed API response.",
            raw_response=response,
        ) from exc

    if not isinstance(text, str) or not text.strip():
        raise ModelResponseError(
            f"{model_name} returned no usable content.",
            raw_response=response,
        )
    return grader.MODEL, text


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned[:60] or "concept_map"


def run_evaluation(
    pdf_path: Path,
    model_names: Iterable[str],
    original_filename: str,
) -> list[EvaluationOutcome]:
    """Grade an uploaded PDF with each selected model and persist outcomes.

    Model-specific failures are returned as EvaluationFailure objects so a
    partial run can still show successful results from other models.
    """
    names = list(model_names)
    if not names:
        raise GradingError("Select at least one model.")
    unknown = [name for name in names if name not in MODEL_CONFIGS]
    if unknown:
        raise GradingError("Unknown model(s): " + ", ".join(unknown))

    _ensure_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex[:8]
    file_stem = _safe_stem(original_filename)
    results: list[EvaluationOutcome] = []

    for model_name in names:
        model_id = MODEL_CONFIGS[model_name]["model_id"]
        grader = GRADER_MODULES[model_name]
        image = render_pdf_image(pdf_path, model_name)
        prompt = grader.build_prompt(Path(original_filename).name)
        raw_text: str | None = None
        try:
            returned_model_id, raw_text = _request_model(model_name, prompt, image)
            cleaned_text = grader.clean_json_output(raw_text)
            data = parse_model_json(cleaned_text)
            output_path = OUTPUT_DIR / (
                f"{timestamp}_{run_id}_{file_stem}_{model_name.lower()}.json"
            )
            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            results.append(
                EvaluationResult(model_name, returned_model_id, data, output_path)
            )
        except (ModelResponseError, MalformedResultError) as exc:
            raw_response = getattr(exc, "raw_response", None)
            if raw_response is None:
                raw_response = raw_text
            debug_path = _save_failed_response(
                timestamp=timestamp,
                run_id=run_id,
                file_stem=file_stem,
                model_name=model_name,
                model_id=model_id,
                error_message=str(exc),
                raw_response=raw_response,
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
