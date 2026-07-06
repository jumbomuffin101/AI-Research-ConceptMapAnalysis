"""Upload-driven grading pipeline used by the Streamlit demo.

The command-line graders remain independent. This module mirrors their proven
PDF rendering flow while accepting any uploaded PDF.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = PROJECT_ROOT / "rubric" / "concept_map_rubric.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web_demo"
DEBUG_DIR = OUTPUT_DIR / "debug"

GEMMA_MODEL = "google/gemma-4-26b-a4b-it:free"
NEMOTRON_MODEL = "meta/llama-3.2-90b-vision-instruct"

GRADER_MODULES = {
    "Gemma": None,
    "Llama": None,
}

MODEL_CONFIGS = {
    "Gemma": {
        "model_id": GEMMA_MODEL,
        "max_tokens": 2000,
    },
    "Llama": {
        "model_id": NEMOTRON_MODEL,
        "max_tokens": 2500,
    },
}

MODEL_PROVIDER_INFO = {
    "Gemma": {
        "provider": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": GEMMA_MODEL,
    },
    "Llama": {
        "provider": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": NEMOTRON_MODEL,
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

DOMAIN_OVERALL_QUESTIONS = {
    "knowledge_acquisition": (
        "Does the student's map include key knowledge from the case and content "
        "learned during this unit?"
    ),
    "integration": "Did the learner connect key knowledge accurately & comprehensively?",
    "application": "Did the learner explain key clinical data with relevant basic science?",
    "transfer": "Did the learner use previously learned content to deepen understanding?",
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
    normalized = selection.strip()
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
    """Return provider/model debug lines safe for display or logs."""
    if model_names is None:
        return [
            "Gemma provider: OpenRouter",
            f"Gemma model: {GEMMA_MODEL}",
            "Llama provider: NVIDIA NIM",
            f"Llama model: {NEMOTRON_MODEL}",
        ]

    lines = []
    for model_name in model_names:
        info = MODEL_PROVIDER_INFO.get(model_name)
        if not info:
            continue
        lines.append(
            f"{model_name} provider: {info['provider']} | "
            f"base_url: {info['base_url']} | model: {info['model']}"
        )
    return lines


def render_pdf_image(pdf_path: Path, model_name: str) -> str:
    """Render the uploaded PDF as a deployment-safe base64 image."""
    _ = model_name
    try:
        import fitz
    except ImportError as exc:
        raise GradingError(
            "PyMuPDF is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    try:
        with fitz.open(pdf_path) as document:
            if document.page_count < 1:
                raise InvalidPDFError("The uploaded PDF has no pages.")
            page = document[0]
            scale = 2.0 if model_name == "Llama" else 1.0
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            image_bytes = (
                pixmap.tobytes("jpeg", jpg_quality=80)
                if model_name == "Llama"
                else pixmap.tobytes("png")
            )
            return base64.b64encode(image_bytes).decode("utf-8")
    except InvalidPDFError:
        raise
    except (
        fitz.FileDataError,
        fitz.EmptyFileError,
        RuntimeError,
        ValueError,
        IndexError,
    ) as exc:
        raise InvalidPDFError("The uploaded file is not a valid, readable PDF.") from exc


def load_summative_rubric() -> dict[str, Any]:
    """Load the Spring 2025 summative rubric used by grading prompts."""
    try:
        rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GradingError(f"Rubric not found at {RUBRIC_PATH}.") from exc
    except json.JSONDecodeError as exc:
        raise GradingError("The rubric file contains invalid JSON.") from exc

    return {
        group: rubric[group]
        for group in CATEGORY_FIELDS
        if isinstance(rubric.get(group), dict)
    }


def build_spring_schema(map_file: str, model_id: str) -> dict[str, Any]:
    """Build the Spring 2025 summative grading JSON shape."""
    schema: dict[str, Any] = {"map_file": map_file, "model": model_id}
    for group, fields in CATEGORY_FIELDS.items():
        schema[group] = {
            field: {
                "score": 1,
                "explanation": "",
                "evidence_from_map": [],
            }
            for field in fields
        }
        schema[group]["overall_decision"] = "No"
        schema[group]["if_no_explanation"] = ""
    schema["overall_meets_expectations"] = "No"
    schema["strengths"] = ["", ""]
    schema["areas_for_improvement"] = ["", ""]
    schema["grading_notes"] = ""
    return schema


def build_web_prompt(map_file: str, model_id: str) -> str:
    """Build the shorter Streamlit-compatible grading prompt."""
    schema = build_spring_schema(map_file, model_id)
    rubric = load_summative_rubric()

    return f"""Use the Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities exactly.
Do not invent additional grading criteria.

Rubric:
{json.dumps(rubric, indent=2)}

Global rules:
- Every criterion score must be an integer 1, 2, 3, or 4 only.
- Every domain overall_decision must be exactly "Yes" or "No".
- overall_meets_expectations must be exactly "Yes" or "No".
- Do not output Partial, Partially Meets, Borderline, Maybe, score 0, score 5, decimal scores, or any score outside 1-4.
- If evidence is missing, write "No clear evidence found in the concept map."
- Do not hallucinate evidence not visible in the concept map.

For every scored category, return only:
- score: integer 1-4
- explanation: one short explanation
- evidence_from_map: short strings copied or paraphrased from visible map content

Each domain must include:
- overall_decision: "Yes" or "No"
- if_no_explanation: required when overall_decision is "No"; otherwise empty string

Keep all JSON string values short. Do not write paragraphs.
Return JSON only. Do not include markdown or text outside JSON.
Use this exact JSON structure:
{json.dumps(schema, indent=2)}
"""


def build_model_prompt(model_name: str, map_file: str, model_id: str) -> str:
    """Build full-schema prompts while keeping Gemma's prompt unchanged."""
    if model_name != "Llama":
        return build_web_prompt(map_file, model_id)

    rubric = json.dumps(load_summative_rubric(), separators=(",", ":"))
    schema = json.dumps(
        build_spring_schema(map_file, model_id), separators=(",", ":")
    )
    return f"""Use the Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities exactly. Do not invent criteria.
Rubric:{rubric}
Rules: criterion scores are integers 1-4 only; domain overall_decision and overall_meets_expectations are exactly Yes or No; never use Partial, decimals, 0, or 5. Each criterion needs score, one short explanation, and brief evidence_from_map from visible content. Do not hallucinate. Each domain needs overall_decision and if_no_explanation when No. Keep strengths, areas_for_improvement, and grading_notes brief.
Return ONLY raw valid minified JSON matching this exact schema. No markdown or prose. First character {{; last character }}.
Schema:{schema}
"""


def _strip_json_fences(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$", "", text)


def _extract_first_complete_json_object(text: str) -> str | None:
    """Return the first balanced JSON object substring, if one exists."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_repairable_json_object(text: str) -> str | None:
    """Extract JSON and append missing closers only when structurally obvious."""
    start = text.find("{")
    if start < 0:
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    last_index = start

    for index in range(start, len(text)):
        char = text[index]
        last_index = index
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]

    if in_string or escaped or not stack:
        return None

    candidate = text[start : last_index + 1].rstrip()
    candidate = re.sub(r",\s*$", "", candidate)
    return candidate + "".join(reversed(stack))


def _load_json_with_repair(raw_text: str) -> dict[str, Any]:
    """Parse JSON, then fall back to the first complete object if needed."""
    text = _strip_json_fences(raw_text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as first_exc:
        candidate = _extract_first_complete_json_object(text)
        if candidate is None:
            candidate = _extract_repairable_json_object(text)
        if candidate is None:
            raise MalformedResultError(
                "The model returned malformed JSON "
                f"({first_exc.msg}, line {first_exc.lineno})."
            ) from first_exc
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError as second_exc:
            raise MalformedResultError(
                "The model returned malformed JSON "
                f"({second_exc.msg}, line {second_exc.lineno})."
            ) from second_exc

    if not isinstance(result, dict):
        raise MalformedResultError("The model result must be a JSON object.")
    return result


def _cleaned_json_text(raw_text: str) -> str:
    """Return the JSON text selected by the same repair path used for parsing."""
    text = _strip_json_fences(raw_text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return (
            _extract_first_complete_json_object(text)
            or _extract_repairable_json_object(text)
            or text
        )


def _contains_forbidden_decision_text(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_forbidden_decision_text(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_decision_text(item) for item in value)
    if not isinstance(value, str):
        return False
    return any(
        re.search(rf"\b{re.escape(term)}\b", value, flags=re.IGNORECASE)
        for term in FORBIDDEN_DECISION_TEXT
    )


def _require_yes_no(value: Any, field_path: str) -> None:
    if value not in {"Yes", "No"}:
        raise MalformedResultError(f"'{field_path}' must be exactly 'Yes' or 'No'.")


def _fill_missing_domain_explanations(result: dict[str, Any]) -> None:
    """Supply the schema fallback for unexplained negative domain decisions."""
    fallback = "The model did not provide a domain-level explanation."
    for group in CATEGORY_FIELDS:
        section = result.get(group)
        if not isinstance(section, dict) or section.get("overall_decision") != "No":
            continue
        explanation = section.get("if_no_explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            section["if_no_explanation"] = fallback


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Extract, parse, repair when possible, and validate grading JSON."""
    if not raw_text or not raw_text.strip():
        raise ModelResponseError("The model returned an empty response.")

    if "{" not in raw_text:
        raise MalformedResultError("The model response did not contain a JSON object.")

    result = _load_json_with_repair(raw_text)
    _fill_missing_domain_explanations(result)
    if _contains_forbidden_decision_text(result):
        raise MalformedResultError(
            "The model result contains a forbidden non-binary decision label."
        )

    missing = [
        key
        for key in (*CATEGORY_FIELDS.keys(), "overall_meets_expectations")
        if key not in result
    ]
    if missing:
        raise MalformedResultError(
            "The model result is missing required fields: " + ", ".join(missing)
        )

    _require_yes_no(
        result.get("overall_meets_expectations"),
        "overall_meets_expectations",
    )

    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            raise MalformedResultError(f"'{group}' must be a JSON object.")
        _require_yes_no(section.get("overall_decision"), f"{group}.overall_decision")
        if (
            section.get("overall_decision") == "No"
            and not str(section.get("if_no_explanation", "")).strip()
        ):
            raise MalformedResultError(
                f"'{group}.if_no_explanation' is required when overall_decision is 'No'."
            )
        for field in fields:
            item = section.get(field)
            score = item.get("score") if isinstance(item, dict) else None
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 4:
                raise MalformedResultError(
                    f"'{group}.{field}.score' must be an integer from 1 to 4."
                )
            if not isinstance(item.get("explanation"), str):
                raise MalformedResultError(
                    f"'{group}.{field}.explanation' must be a string."
                )
    return result


def _is_implausible_all_four_result(result: dict[str, Any]) -> bool:
    """Return true only when every required rubric criterion is scored four."""
    scores: list[Any] = []
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            return False
        for field in fields:
            item = section.get(field)
            if not isinstance(item, dict):
                return False
            scores.append(item.get("score"))
    return bool(scores) and all(score == 4 for score in scores)


def _is_implausible_all_one_result(result: dict[str, Any]) -> bool:
    """Return true only when every required rubric criterion is scored one."""
    scores: list[Any] = []
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            return False
        for field in fields:
            item = section.get(field)
            if not isinstance(item, dict):
                return False
            scores.append(item.get("score"))
    return bool(scores) and all(score == 1 for score in scores)


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


def _save_nemotron_debug_image(
    *, image: str, timestamp: str, run_id: str, file_stem: str
) -> Path:
    """Save the exact image bytes included in the NVIDIA request."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(image, validate=True)
    _, extension = _image_media_type(image_bytes)
    image_path = DEBUG_DIR / (
        f"{timestamp}_{run_id}_{file_stem}_llama_request{extension}"
    )
    image_path.write_bytes(image_bytes)
    return image_path


def _save_nemotron_trace(
    *,
    timestamp: str,
    run_id: str,
    file_stem: str,
    map_filename: str,
    pdf_path: Path,
    prompt: str,
    image_path: Path,
    raw_api_response: Any | None,
    cleaned_json: str | None,
    parsed_before_validation: dict[str, Any] | None,
    parsed_after_validation: dict[str, Any] | None,
    request_metadata: dict[str, Any] | None,
    error_message: str | None,
) -> Path:
    """Persist a complete, per-run NVIDIA request/response trace."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    debug_path = DEBUG_DIR / (
        f"{timestamp}_{run_id}_{file_stem}_llama_trace.json"
    )
    payload = {
        "timestamp": timestamp,
        "provider": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model_id": NEMOTRON_MODEL,
        "map_filename": map_filename,
        "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "prompt_text": prompt,
        "prompt_length": len(prompt),
        "image_debug_path": str(image_path),
        "image_file_size": image_path.stat().st_size,
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "request_metadata": request_metadata,
        "raw_api_response": _response_to_debug_text(raw_api_response),
        "cleaned_json": cleaned_json,
        "parsed_json_before_validation": parsed_before_validation,
        "parsed_json_after_validation": parsed_after_validation,
        "error_message": error_message,
    }
    debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return debug_path


def _save_llama_raw_debug(
    *,
    timestamp: str,
    run_id: str,
    file_stem: str,
    attempt: int,
    raw_text: str,
    response: Any,
    prompt: str,
    max_tokens: int,
    image_file_size: int,
) -> Path:
    """Save Llama content and generation metadata before JSON parsing."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    choices = getattr(response, "choices", None)
    finish_reason = (
        getattr(choices[0], "finish_reason", None) if choices else None
    )
    debug_path = DEBUG_DIR / (
        f"{timestamp}_{run_id}_{file_stem}_llama_attempt{attempt}_raw.json"
    )
    debug_path.write_text(
        json.dumps(
            {
                "provider": "NVIDIA NIM",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "model": NEMOTRON_MODEL,
                "raw_text": raw_text,
                "cleaned_text": _strip_json_fences(raw_text),
                "prompt_length": len(prompt),
                "max_tokens": max_tokens,
                "image_file_size": image_file_size,
                "finish_reason": finish_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return debug_path


def _get_secret(name: str) -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
    except Exception:
        return None
    return str(secret_value) if secret_value else None


def _openai_client(**options: Any) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise GradingError(
            "The OpenAI SDK is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return OpenAI(**options)


def create_openrouter_client(*, disable_sdk_retries: bool = False) -> Any:
    api_key = _get_secret("OPENROUTER_API_KEY")
    if not api_key:
        raise GradingError(
            "OPENROUTER_API_KEY is missing. Add it to the environment, Streamlit secrets, or project .env file."
        )
    options: dict[str, Any] = {
        "api_key": api_key,
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 300,
    }
    if disable_sdk_retries:
        options["max_retries"] = 0
    return _openai_client(
        **options
    )


def create_nvidia_client(*, disable_sdk_retries: bool = False) -> Any:
    api_key = _get_secret("NVIDIA_API_KEY")
    if not api_key:
        raise GradingError("NVIDIA_API_KEY is not configured.")
    options: dict[str, Any] = {
        "api_key": api_key,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "timeout": 300,
    }
    if disable_sdk_retries:
        options["max_retries"] = 0
    return _openai_client(
        **options
    )


def _create_client(
    model_name: str, *, disable_sdk_retries: bool = False
) -> Any:
    if model_name == "Llama":
        return create_nvidia_client(disable_sdk_retries=disable_sdk_retries)
    return create_openrouter_client(disable_sdk_retries=disable_sdk_retries)


def _is_input_limit_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        phrase in lowered
        for phrase in (
            "prompt tokens limit exceeded",
            "input token",
            "context length",
            "maximum context",
            "token limit",
        )
    )


def _image_media_type(image_bytes: bytes) -> tuple[str, str]:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    raise GradingError("The rendered request image is not a valid JPEG or PNG.")


def _prepare_request_image(
    model_name: str, prompt: str, image: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build NVIDIA's documented nested image_url message content."""
    image_bytes = base64.b64decode(image, validate=True)
    media_type, _ = _image_media_type(image_bytes)
    request_metadata: dict[str, Any] = {
        "image_bytes": len(image_bytes),
        "image_transport": "inline_base64",
        "payload_shape": {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,<exact bytes saved in image_debug_path>"
            },
        },
    }
    image_item = {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{image}"},
    }
    text_item = {"type": "text", "text": prompt}
    content = [image_item, text_item] if model_name == "Llama" else [
        text_item,
        image_item,
    ]
    return content, request_metadata


def _write_health_debug(
    path: Path,
    *,
    payload_shape: dict[str, Any],
    response: Any = None,
    error: Any = None,
    prompt_length: int = 0,
    image_file_size: int = 0,
    extracted_terms: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "provider": "NVIDIA NIM",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "model_id": NEMOTRON_MODEL,
                "prompt_length": prompt_length,
                "image_file_size": image_file_size,
                "payload_shape": payload_shape,
                "raw_response": _response_to_debug_text(response),
                "extracted_visible_terms": extracted_terms,
                "error": str(error) if error else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _require_response_content(response: Any, test_name: str) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ModelResponseError(f"Llama {test_name} returned no response choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError(f"Llama {test_name} returned empty content.")
    return content


def _extract_visible_terms(text: str) -> list[str]:
    candidates = re.split(r"[\r\n|;,]+", text)
    terms: list[str] = []
    for candidate in candidates:
        term = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", candidate).strip(" \t\"'")
        if not term or not re.search(r"[A-Za-z]", term):
            continue
        lowered = term.lower()
        if lowered.startswith(("here are", "visible words", "the image shows")):
            continue
        if term not in terms:
            terms.append(term)
    return terms[:10]


def _extract_preflight_terms(text: str) -> list[str]:
    """Extract the model's explicitly reported visible medical terms."""
    parts = re.split(r"visible[_ ]terms\s*:", text, maxsplit=1, flags=re.IGNORECASE)
    term_text = parts[1] if len(parts) == 2 else text
    ignored = {
        "concept map",
        "medical concept map",
        "image",
        "nodes",
        "labels",
        "relationships",
    }
    terms: list[str] = []
    for candidate in re.split(r"[\r\n|;,]+", term_text):
        term = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", candidate).strip(" \t\"'")
        if (
            not term
            or not re.search(r"[A-Za-z]", term)
            or len(term) > 100
            or term.lower() in ignored
        ):
            continue
        if term not in terms:
            terms.append(term)
    return terms[:10]


def _run_nemotron_health_tests(
    client: Any, image: str, debug_prefix: Path
) -> tuple[str, list[str]]:
    """Gate full grading on text and image calls using NVIDIA's sample format."""
    text_prompt = "Reply with OK."
    text_payload = {
        "model": NEMOTRON_MODEL,
        "messages": [{"role": "user", "content": text_prompt}],
        "temperature": 0,
        "max_tokens": 16,
    }
    text_path = Path(f"{debug_prefix}_text_health.json")
    text_response: Any | None = None
    try:
        text_response = client.chat.completions.create(**text_payload)
        _require_response_content(text_response, "text health test")
        _write_health_debug(
            text_path,
            payload_shape=text_payload,
            response=text_response,
            prompt_length=len(text_prompt),
        )
    except Exception as exc:
        _write_health_debug(
            text_path,
            payload_shape=text_payload,
            response=text_response,
            error=exc,
            prompt_length=len(text_prompt),
        )
        raise ModelResponseError(
            "NVIDIA text endpoint failed", raw_response=text_response or repr(exc)
        ) from exc

    preflight_prompt = (
        "Describe this concept map in one sentence. Then write VISIBLE_TERMS: "
        "followed by up to 10 medical concepts or terms that are clearly visible, "
        "separated by semicolons."
    )
    image_content, image_metadata = _prepare_request_image(
        "Llama", preflight_prompt, image
    )
    image_payload = {
        "model": NEMOTRON_MODEL,
        "messages": [{"role": "user", "content": image_content}],
        "temperature": 0,
        "max_tokens": 100,
    }
    image_payload_shape = {
        **image_payload,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_metadata["payload_shape"],
                    {"type": "text", "text": preflight_prompt},
                ],
            }
        ],
    }
    image_path = Path(f"{debug_prefix}_image_health.json")
    image_response: Any | None = None
    image_file_size = image_metadata["image_bytes"]
    try:
        image_response = client.chat.completions.create(**image_payload)
        preflight_text = _require_response_content(image_response, "image preflight")
        visible_terms = _extract_preflight_terms(preflight_text)
        _write_health_debug(
            image_path,
            payload_shape=image_payload_shape,
            response=image_response,
            prompt_length=len(preflight_prompt),
            image_file_size=image_file_size,
            extracted_terms=visible_terms,
        )
        return preflight_text.strip(), visible_terms
    except Exception as exc:
        _write_health_debug(
            image_path,
            payload_shape=image_payload_shape,
            response=image_response,
            error=exc,
            prompt_length=len(preflight_prompt),
            image_file_size=image_file_size,
        )
        raise ModelResponseError(
            "NVIDIA image input failed", raw_response=image_response or repr(exc)
        ) from exc


NEMOTRON_EVIDENCE_FIELDS = (
    "visible_diagnoses",
    "patient_data",
    "basic_science_concepts",
    "clinical_science_concepts",
    "health_system_science_concepts",
    "determinants_of_health",
    "visible_relationships_connections",
    "missing_or_unclear_required_elements",
)


def _build_nemotron_evidence_prompt() -> str:
    evidence_schema = {field: [] for field in NEMOTRON_EVIDENCE_FIELDS}
    return f"""Extract only evidence visibly present in this concept map image.
Do not grade the map. Do not infer from medical knowledge or add content that is not visible.
Extract visible diagnoses, patient data, basic science concepts, clinical science concepts,
health system science concepts, determinants of health, and visible relationships or connections.
Also identify required elements that are visibly missing or unclear within those categories.
Use short strings. If a category has no visible evidence, return an empty list.

Return ONLY valid JSON with this exact structure:
{json.dumps(evidence_schema, indent=2)}
"""


def _parse_nemotron_evidence(raw_text: str) -> dict[str, list[str]]:
    evidence = _load_json_with_repair(raw_text)
    missing = [field for field in NEMOTRON_EVIDENCE_FIELDS if field not in evidence]
    if missing:
        raise MalformedResultError(
            "Nemotron evidence extraction is missing fields: " + ", ".join(missing)
        )
    parsed: dict[str, list[str]] = {}
    for field in NEMOTRON_EVIDENCE_FIELDS:
        value = evidence[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise MalformedResultError(
                f"Nemotron evidence field '{field}' must be a list of strings."
            )
        parsed[field] = value
    return parsed


NEMOTRON_COMPACT_LENGTHS = {"ka": 5, "int": 5, "app": 2, "tr": 3}
NEMOTRON_COMPACT_OVERALLS = (
    "ka_overall",
    "int_overall",
    "app_overall",
    "tr_overall",
    "overall",
)
NEMOTRON_SCORE_EXPLANATIONS = {
    1: "Little or irrelevant evidence was identified for this criterion.",
    2: "Evidence was partly relevant, too general, or limited.",
    3: "Evidence was relevant and mostly synthesized.",
    4: "Evidence was synthesized, detailed, and well-supported.",
}
NEMOTRON_DOMAIN_MAP = {
    "knowledge_acquisition": ("ka", "ka_overall", "ka", "Knowledge Acquisition"),
    "integration": ("int", "int_overall", "int", "Integration"),
    "application": ("app", "app_overall", "app", "Application"),
    "transfer": ("tr", "tr_overall", "tr", "Transfer"),
}


def _build_nemotron_compact_prompt(evidence: dict[str, list[str]]) -> str:
    compact_schema = {
        "ka": [1, 1, 1, 1, 1],
        "int": [1, 1, 1, 1, 1],
        "app": [1, 1],
        "tr": [1, 1, 1],
        "ka_overall": "No",
        "int_overall": "No",
        "app_overall": "No",
        "tr_overall": "No",
        "overall": "No",
        "evidence": {"ka": [], "int": [], "app": [], "tr": []},
        "improvements": [],
    }
    return f"""Independently grade the complete Spring 2025 concept map rubric using only the extracted visible evidence.
Do not infer from medical knowledge or grade what should be present.
Assign every criterion score as an integer 1-4 and every overall value as exactly "Yes" or "No".
The score arrays must follow the rubric criterion order shown below.
Keep at most one short evidence item per domain and at most two short improvements. Return compact minified JSON only. No explanations, markdown, or extra text.

Spring 2025 rubric:
{json.dumps(load_summative_rubric(), separators=(",", ":"))}

Extracted evidence:
{json.dumps(evidence, separators=(",", ":"))}

Exact compact JSON structure:
{json.dumps(compact_schema, separators=(",", ":"))}
"""


def _validate_nemotron_compact(data: dict[str, Any]) -> None:
    for key, expected_length in NEMOTRON_COMPACT_LENGTHS.items():
        scores = data.get(key)
        if not isinstance(scores, list) or len(scores) != expected_length:
            raise MalformedResultError(
                f"Nemotron compact field '{key}' must contain {expected_length} scores."
            )
        if any(
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 1 <= score <= 4
            for score in scores
        ):
            raise MalformedResultError(
                f"Nemotron compact field '{key}' must contain integers from 1 to 4."
            )
    for key in NEMOTRON_COMPACT_OVERALLS:
        if data.get(key) not in {"Yes", "No"}:
            raise MalformedResultError(
                f"Nemotron compact field '{key}' must be exactly 'Yes' or 'No'."
            )
    evidence = data.get("evidence", {})
    if evidence is not None and not isinstance(evidence, dict):
        raise MalformedResultError("Nemotron compact 'evidence' must be an object.")
    improvements = data.get("improvements", [])
    if not isinstance(improvements, list) or not all(
        isinstance(item, str) for item in improvements
    ):
        raise MalformedResultError(
            "Nemotron compact 'improvements' must be a list of strings."
        )


def _compact_evidence(data: dict[str, Any], key: str) -> list[str]:
    evidence = data.get("evidence")
    values = evidence.get(key) if isinstance(evidence, dict) else None
    if isinstance(values, list):
        cleaned = [item.strip() for item in values if isinstance(item, str) and item.strip()]
        if cleaned:
            return cleaned
    return ["No clear evidence found in the concept map."]


def _expand_nemotron_compact(
    data: dict[str, Any], map_file: str
) -> dict[str, Any]:
    result = build_spring_schema(map_file, NEMOTRON_MODEL)
    strengths: list[str] = []
    for group, (score_key, overall_key, evidence_key, label) in NEMOTRON_DOMAIN_MAP.items():
        domain = result[group]
        domain_evidence = _compact_evidence(data, evidence_key)
        for field, score in zip(CATEGORY_FIELDS[group], data[score_key]):
            domain[field] = {
                "score": score,
                "explanation": NEMOTRON_SCORE_EXPLANATIONS[score],
                "evidence_from_map": list(domain_evidence),
            }
        decision = data[overall_key]
        domain["overall_decision"] = decision
        domain["if_no_explanation"] = (
            f"{label} is marked No because Nemotron identified insufficient visible evidence for this domain."
            if decision == "No"
            else ""
        )
        if decision == "Yes" and domain_evidence[0] != "No clear evidence found in the concept map.":
            strengths.append(domain_evidence[0])
    result["overall_meets_expectations"] = data["overall"]
    result["strengths"] = strengths[:2]
    result["areas_for_improvement"] = data.get("improvements", [])
    result["grading_notes"] = ""
    return result


def _run_nemotron_compact_grading(
    client: Any,
    image: str,
    map_file: str,
    debug_prefix: Path,
    request_timeout: float | None,
) -> tuple[str, Any, dict[str, Any]]:
    evidence_prompt = _build_nemotron_evidence_prompt()
    evidence_content, image_metadata = _prepare_request_image(
        "Nemotron", evidence_prompt, image
    )
    evidence_options: dict[str, Any] = {
        "model": NEMOTRON_MODEL,
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": evidence_content}],
    }
    if request_timeout is not None:
        evidence_options["timeout"] = request_timeout
    evidence_response = client.chat.completions.create(**evidence_options)
    evidence_text = _require_response_content(evidence_response, "evidence extraction")
    Path(f"{debug_prefix}_evidence_raw_response.json").write_text(
        _response_to_debug_text(evidence_response), encoding="utf-8"
    )
    evidence = _parse_nemotron_evidence(evidence_text)
    evidence_path = Path(f"{debug_prefix}_extracted_evidence.json")
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    compact_prompt = _build_nemotron_compact_prompt(evidence)
    compact_prompt_path = Path(f"{debug_prefix}_compact_grading_prompt.txt")
    compact_prompt_path.write_text(compact_prompt, encoding="utf-8")
    compact_options: dict[str, Any] = {
        "model": NEMOTRON_MODEL,
        "temperature": 0,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": compact_prompt}],
    }
    if request_timeout is not None:
        compact_options["timeout"] = request_timeout
    compact_response = client.chat.completions.create(**compact_options)
    compact_text = _require_response_content(compact_response, "compact grading")
    compact_raw_path = Path(f"{debug_prefix}_compact_grading_raw_response.json")
    compact_raw_path.write_text(
        _response_to_debug_text(compact_response), encoding="utf-8"
    )
    malformed_path: Path | None = None
    retry_prompt_path: Path | None = None
    retry_raw_path: Path | None = None
    try:
        compact_data = _load_json_with_repair(compact_text)
        _validate_nemotron_compact(compact_data)
    except MalformedResultError:
        malformed_path = Path(f"{debug_prefix}_compact_malformed_raw_response.json")
        malformed_path.write_text(
            _response_to_debug_text(compact_response), encoding="utf-8"
        )
        retry_prompt = (
            f"{compact_prompt}\n\nReturn compact minified JSON only. "
            "No explanations. No markdown. No extra text."
        )
        retry_prompt_path = Path(f"{debug_prefix}_compact_retry_prompt.txt")
        retry_prompt_path.write_text(retry_prompt, encoding="utf-8")
        retry_options: dict[str, Any] = {
            "model": NEMOTRON_MODEL,
            "temperature": 0,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": retry_prompt}],
        }
        if request_timeout is not None:
            retry_options["timeout"] = request_timeout
        compact_response = client.chat.completions.create(**retry_options)
        compact_text = _require_response_content(compact_response, "compact grading retry")
        retry_raw_path = Path(f"{debug_prefix}_compact_retry_raw_response.json")
        retry_raw_path.write_text(
            _response_to_debug_text(compact_response), encoding="utf-8"
        )
        compact_data = _load_json_with_repair(compact_text)
        _validate_nemotron_compact(compact_data)
    compact_json_path = Path(f"{debug_prefix}_compact_grading.json")
    compact_json_path.write_text(json.dumps(compact_data, indent=2), encoding="utf-8")
    expanded = _expand_nemotron_compact(compact_data, map_file)
    expanded_text = json.dumps(expanded, separators=(",", ":"))
    metadata = {
        "pipeline": "nemotron_evidence_then_compact_grading_local_expansion",
        "image_bytes": image_metadata["image_bytes"],
        "image_transport": image_metadata["image_transport"],
        "evidence_payload_shape": {
            "model": NEMOTRON_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_metadata["payload_shape"],
                        {"type": "text", "text": evidence_prompt},
                    ],
                }
            ],
        },
        "extracted_evidence_path": str(evidence_path),
        "compact_prompt_path": str(compact_prompt_path),
        "compact_raw_response_path": str(compact_raw_path),
        "compact_json_path": str(compact_json_path),
        "compact_malformed_raw_response_path": (
            str(malformed_path) if malformed_path else None
        ),
        "compact_retry_prompt_path": (
            str(retry_prompt_path) if retry_prompt_path else None
        ),
        "compact_retry_raw_response_path": (
            str(retry_raw_path) if retry_raw_path else None
        ),
    }
    return expanded_text, compact_response, metadata


NEMOTRON_PLAIN_SECTIONS = {
    "knowledge_acquisition": "Knowledge Acquisition",
    "integration": "Integration",
    "application": "Application",
    "transfer": "Transfer",
}
NEMOTRON_SECTION_ALIASES = {
    "knowledge_acquisition": ("Knowledge Acquisition", "knowledge_acquisition", "KA"),
    "integration": ("Integration",),
    "application": ("Application",),
    "transfer": ("Transfer",),
    "final": ("Final", "Overall"),
}


def _build_nemotron_plain_text_prompt() -> str:
    lines = [
        "Independently grade every Spring 2025 rubric criterion using only the extracted visible evidence.",
        "Do not infer from medical knowledge or grade what should be present.",
        "Return plain text only, not JSON or markdown. Use exactly these headers and field names.",
        "Every score must be an integer 1-4. Every decision must be Yes or No. Keep each reason and list item short.",
        "",
    ]
    for group, header in NEMOTRON_PLAIN_SECTIONS.items():
        lines.append(f"{header}:")
        lines.extend(f"{field}: score 1-4" for field in CATEGORY_FIELDS[group])
        lines.extend(["overall_decision: Yes/No", "reason: short reason", ""])
    lines.extend(
        [
            "Final:",
            "overall_meets_expectations: Yes/No",
            "strengths: short item | short item",
            "areas_for_improvement: short item | short item | short item",
            "",
            "Spring 2025 rubric:",
            json.dumps(load_summative_rubric(), separators=(",", ":")),
        ]
    )
    return "\n".join(lines)


def _heading_pattern(aliases: tuple[str, ...]) -> str:
    variants = []
    for alias in aliases:
        variants.append(re.escape(alias).replace(r"\ ", r"[\s_]+"))
    return rf"(?im)^\s*(?:\#{{1,6}}\s*)?(?:\*\*)?(?:{'|'.join(variants)})\s*:?(?:\*\*)?\s*$"


def _plain_section(text: str, section_key: str) -> str | None:
    headings: list[tuple[int, int, str]] = []
    for key, aliases in NEMOTRON_SECTION_ALIASES.items():
        headings.extend(
            (match.start(), match.end(), key)
            for match in re.finditer(_heading_pattern(aliases), text)
        )
    headings.sort()
    for index, (_, end, key) in enumerate(headings):
        if key != section_key:
            continue
        next_start = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        return text[end:next_start]
    return None


def _plain_score(section: str, field: str) -> int:
    field_pattern = re.escape(field).replace("_", r"[\s_-]+")
    match = re.search(
        rf"(?im)^\s*{field_pattern}\s*:\s*(?:score\s*)?(-?\d+(?:\.\d+)?)\s*$",
        section,
    )
    if not match:
        raise MalformedResultError(f"Nemotron score '{field}' is missing.")
    value = match.group(1)
    if not re.fullmatch(r"[1-4]", value):
        raise MalformedResultError(f"Nemotron score '{field}' must be an integer from 1 to 4.")
    return int(value)


def _plain_decision(section: str, field: str, notes: list[str]) -> str:
    match = re.search(
        rf"(?im)^\s*{re.escape(field)}\s*:\s*(Yes|No)\s*$", section
    )
    if match:
        return match.group(1)
    notes.append(f"Nemotron omitted {field}; it defaulted to No.")
    return "No"


def _plain_reason(section: str) -> str:
    match = re.search(r"(?im)^\s*reason\s*:\s*(.+?)\s*$", section)
    return match.group(1).strip() if match else ""


def _plain_items(section: str, field: str) -> list[str]:
    match = re.search(rf"(?im)^\s*{re.escape(field)}\s*:\s*(.*?)\s*$", section)
    if not match or not match.group(1).strip():
        return []
    return [item.strip(" -\t") for item in re.split(r"\s*[|;]\s*", match.group(1)) if item.strip(" -\t")]


def _specific_nemotron_reason(reason: str, group: str | None = None) -> bool:
    normalized = reason.strip().lower()
    generic = {
        "meets expectations",
        "does not meet expectations",
        "insufficient evidence",
        "good",
        "complete",
        "all criteria met",
        "no clear evidence found in the concept map.",
    }
    if len(reason.split()) < 6 or normalized in generic:
        return False
    domain_terms = {
        "knowledge_acquisition": ("science", "patient", "case", "data", "determinant", "diagnosis", "health"),
        "integration": ("connect", "link", "relationship", "differential", "illness", "patient data", "science"),
        "application": ("pathophysiology", "diagnosis", "patient data", "clinical", "science"),
        "transfer": ("prior", "learned", "transfer", "clinical", "basic science", "understanding"),
    }
    return group is None or any(term in normalized for term in domain_terms[group])


def _parse_nemotron_plain_text(text: str, map_file: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ModelResponseError("Nemotron returned empty plain-text grading output.")
    rubric = load_summative_rubric()
    result = build_spring_schema(map_file, NEMOTRON_MODEL)
    notes: list[str] = []
    global_decisions = re.findall(
        r"(?im)^\s*overall_decision\s*:\s*(Yes|No)\s*$", text
    )
    global_reasons = re.findall(r"(?im)^\s*reason\s*:\s*(.+?)\s*$", text)
    for index, group in enumerate(NEMOTRON_PLAIN_SECTIONS):
        section = _plain_section(text, group)
        reason = _plain_reason(section) if section is not None else ""
        if not reason and index < len(global_reasons):
            reason = global_reasons[index].strip()
        evidence = (
            [reason]
            if _specific_nemotron_reason(reason, group)
            else ["No clear evidence found in the concept map."]
        )
        for field in CATEGORY_FIELDS[group]:
            score = _plain_score(text, field)
            descriptor = str(rubric[group][field][str(score)]).rstrip(".")
            result[group][field] = {
                "score": score,
                "explanation": f"Score {score}: {descriptor}.",
                "evidence_from_map": list(evidence),
            }
        decision_match = (
            re.search(r"(?im)^\s*overall_decision\s*:\s*(Yes|No)\s*$", section)
            if section is not None
            else None
        )
        if decision_match:
            decision = decision_match.group(1)
        elif index < len(global_decisions):
            decision = global_decisions[index]
        else:
            decision = "No"
            notes.append(f"Nemotron omitted {group}.overall_decision; it defaulted to No.")
        result[group]["overall_decision"] = decision
        result[group]["if_no_explanation"] = (
            reason
            if decision == "No" and _specific_nemotron_reason(reason, group)
            else (
                "The model did not provide a domain-level explanation."
                if decision == "No"
                else ""
            )
        )
    final_section = _plain_section(text, "final") or text
    result["overall_meets_expectations"] = _plain_decision(
        final_section, "overall_meets_expectations", notes
    )
    result["strengths"] = _plain_items(final_section, "strengths")[:2]
    result["areas_for_improvement"] = _plain_items(
        final_section, "areas_for_improvement"
    )[:3]
    result["grading_notes"] = " ".join(notes)
    return result


def _all_four_reasons_are_specific(result: dict[str, Any]) -> bool:
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group, {})
        first_item = section.get(fields[0], {}) if isinstance(section, dict) else {}
        evidence = first_item.get("evidence_from_map") if isinstance(first_item, dict) else None
        if not isinstance(evidence, list) or not evidence or not _specific_nemotron_reason(str(evidence[0]), group):
            return False
    return True


def _run_nemotron_plain_text_grading(
    client: Any,
    image: str,
    map_file: str,
    debug_prefix: Path,
    request_timeout: float | None,
) -> tuple[str, Any, dict[str, Any]]:
    plain_prompt = _build_nemotron_plain_text_prompt()
    prompt_path = Path(f"{debug_prefix}_plain_grading_prompt.txt")
    prompt_path.write_text(plain_prompt, encoding="utf-8")
    plain_content, image_metadata = _prepare_request_image(
        "Nemotron", plain_prompt, image
    )
    options: dict[str, Any] = {
        "model": NEMOTRON_MODEL,
        "temperature": 0,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": plain_content}],
    }
    if request_timeout is not None:
        options["timeout"] = request_timeout
    response = client.chat.completions.create(**options)
    plain_text = _require_response_content(response, "plain-text rubric grading")
    raw_path = Path(f"{debug_prefix}_plain_grading_raw_response.txt")
    raw_path.write_text(plain_text, encoding="utf-8")
    retry_raw_path: Path | None = None
    try:
        parsed = _parse_nemotron_plain_text(plain_text, map_file)
    except MalformedResultError:
        retry_prompt = (
            "Use the exact headings and field names shown below. Do not rename, omit, or reformat them.\n\n"
            f"{plain_prompt}"
        )
        retry_content, _ = _prepare_request_image("Nemotron", retry_prompt, image)
        retry_options = dict(options)
        retry_options["messages"] = [{"role": "user", "content": retry_content}]
        retry_response = client.chat.completions.create(**retry_options)
        retry_text = _require_response_content(
            retry_response, "plain-text rubric grading retry"
        )
        retry_raw_path = Path(f"{debug_prefix}_plain_grading_retry_raw_response.txt")
        retry_raw_path.write_text(retry_text, encoding="utf-8")
        try:
            parsed = _parse_nemotron_plain_text(retry_text, map_file)
        except MalformedResultError as exc:
            raise ModelResponseError(
                "Nemotron plain-text grading could not be parsed after one retry.",
                raw_response=json.dumps(
                    {"initial_plain_text": plain_text, "retry_plain_text": retry_text}
                ),
            ) from exc
        response = retry_response
        plain_text = retry_text
    parsed_path = Path(f"{debug_prefix}_plain_parsed_grading.json")
    parsed_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    metadata = {
        "pipeline": "nemotron_plain_text_grading_local_expansion",
        "image_bytes": image_metadata["image_bytes"],
        "image_transport": image_metadata["image_transport"],
        "plain_prompt_path": str(prompt_path),
        "plain_raw_response_path": str(raw_path),
        "plain_retry_raw_response_path": (
            str(retry_raw_path) if retry_raw_path else None
        ),
        "plain_parsed_grading_path": str(parsed_path),
    }
    return json.dumps(parsed, separators=(",", ":")), response, metadata


def _request_model(
    model_name: str,
    prompt: str,
    image: str,
    max_tokens: int | None = None,
    request_timeout: float | None = None,
    health_debug_prefix: Path | None = None,
    map_file: str | None = None,
    perform_health_test: bool = True,
) -> tuple[str, str, Any, dict[str, Any]]:
    config = MODEL_CONFIGS[model_name]

    try:
        client = _create_client(
            model_name, disable_sdk_retries=request_timeout is not None
        )
        if model_name == "Llama" and perform_health_test:
            for line in model_debug_lines([model_name]):
                print(line)
            if health_debug_prefix is None:
                raise GradingError("Llama health-test debug path is missing.")
            if map_file is None:
                raise GradingError("Llama map filename is missing.")
            preflight_description, visible_terms = _run_nemotron_health_tests(
                client, image, health_debug_prefix
            )
            prompt += (
                "\nImage preflight description: "
                + preflight_description
                + "\nExtracted visible medical terms: "
                + ("; ".join(visible_terms) if visible_terms else "None identified")
                + "\nCalibration rules: Use visible nodes, labels, and relationships as evidence. "
                "Do not require perfect OCR. Do not assign score 1 when relevant visible content exists. "
                "Score 1 only when the criterion is absent, irrelevant, or unreadable. "
                "Score 2 when content is visible but too general or weakly connected. "
                "Score 3 when content is relevant and mostly synthesized. "
                "Score 4 only when content is detailed, comprehensive, and clearly connected. "
                "Do not hallucinate evidence.\n"
            )

        content, request_metadata = _prepare_request_image(
            model_name, prompt, image
        )
        if model_name == "Llama" and perform_health_test:
            request_metadata["preflight_description"] = preflight_description
            request_metadata["visible_terms"] = visible_terms
            request_metadata["preflight_debug_path"] = str(
                Path(f"{health_debug_prefix}_image_health.json")
            )
            request_metadata["effective_prompt"] = prompt
        request_options: dict[str, Any] = {
            "model": config["model_id"],
            "max_tokens": max_tokens or config["max_tokens"],
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        payload_shape = [
            {"type": "text", "text": prompt},
            request_metadata["payload_shape"],
        ]
        if model_name == "Llama":
            payload_shape.reverse()
        request_metadata["outgoing_payload_shape"] = {
            "model": config["model_id"],
            "max_tokens": request_options["max_tokens"],
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": payload_shape,
                }
            ],
        }
        if request_timeout is not None:
            request_options["timeout"] = request_timeout
        response = client.chat.completions.create(
            **request_options
        )
    except Exception as exc:
        message = str(exc)
        if isinstance(exc, ModelResponseError) and message in {
            "NVIDIA text endpoint failed",
            "NVIDIA image input failed",
        }:
            raise
        if "NVCF asset pool must be given" in message:
            message = (
                "NVIDIA NIM rejected the image payload format. The request likely "
                "needs NVIDIA's asset upload/image format instead of the current "
                "base64 image_url."
            )
        if _is_input_limit_error(message):
            message = (
                "Input is too large for the current model limit. "
                "Try a smaller PDF/image or use the local CLI pipeline."
            )
        if model_name == "Llama":
            raise ModelResponseError(
                f"NVIDIA grading request failed: {message}",
                raw_response=getattr(exc, "raw_response", repr(exc)),
            ) from exc
        provider = "OpenRouter"
        raise ModelResponseError(
            f"{model_name} {provider} API request failed: {message}",
            raw_response=getattr(exc, "raw_response", repr(exc)),
        ) from exc

    choices = getattr(response, "choices", None)
    if not choices:
        if model_name == "Llama":
            raise ModelResponseError(
                "NVIDIA grading request failed: no response choices.",
                raw_response=response,
            )
        raise ModelResponseError(
            f"{model_name} returned no response choices.",
            raw_response=response,
        )
    try:
        text = choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        if model_name == "Llama":
            raise ModelResponseError(
                "NVIDIA grading request failed: malformed API response.",
                raw_response=response,
            ) from exc
        raise ModelResponseError(
            f"{model_name} returned a malformed API response.",
            raw_response=response,
        ) from exc

    if not isinstance(text, str) or not text.strip():
        if model_name == "Llama":
            raise ModelResponseError(
                "NVIDIA grading request failed: empty content.", raw_response=response
            )
        raise ModelResponseError(
            f"{model_name} returned no usable content.",
            raw_response=response,
        )
    return config["model_id"], text, response, request_metadata


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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex[:8]
    file_stem = _safe_stem(original_filename)
    results: list[EvaluationOutcome] = []

    for model_name in names:
        model_id = MODEL_CONFIGS[model_name]["model_id"]
        image = render_pdf_image(pdf_path, model_name)
        raw_text: str | None = None
        raw_api_response: Any | None = None
        prompt = ""
        image_debug_path: Path | None = None
        cleaned_json: str | None = None
        parsed_before_validation: dict[str, Any] | None = None
        parsed_after_validation: dict[str, Any] | None = None
        request_metadata: dict[str, Any] | None = None
        preflight_terms: list[str] = []
        llama_attempt = 1
        try:
            prompt = build_model_prompt(
                model_name, Path(original_filename).name, model_id
            )
            if model_name == "Llama":
                image_debug_path = _save_nemotron_debug_image(
                    image=image,
                    timestamp=timestamp,
                    run_id=run_id,
                    file_stem=file_stem,
                )
            (
                returned_model_id,
                raw_text,
                raw_api_response,
                request_metadata,
            ) = _request_model(
                model_name,
                prompt,
                image,
                health_debug_prefix=(
                    DEBUG_DIR
                    / f"{timestamp}_{run_id}_{file_stem}_llama"
                    if model_name == "Llama"
                    else None
                ),
                map_file=(
                    Path(original_filename).name
                    if model_name == "Llama"
                    else None
                ),
            )
            if model_name == "Llama":
                prompt = str(request_metadata.get("effective_prompt", prompt))
                preflight_terms = list(request_metadata.get("visible_terms", []))
                _save_llama_raw_debug(
                    timestamp=timestamp,
                    run_id=run_id,
                    file_stem=file_stem,
                    attempt=1,
                    raw_text=raw_text,
                    response=raw_api_response,
                    prompt=prompt,
                    max_tokens=MODEL_CONFIGS["Llama"]["max_tokens"],
                    image_file_size=image_debug_path.stat().st_size,
                )
            try:
                parsed_before_validation = (
                    _load_json_with_repair(raw_text)
                    if model_name == "Llama"
                    else None
                )
                data = parse_model_json(raw_text)
            except MalformedResultError:
                if model_name != "Llama":
                    raise
                llama_attempt += 1
                retry_prompt = (
                    f"{prompt}\n\nYour previous answer was not valid JSON. "
                    "Return the same evaluation as valid minified JSON only."
                )
                (
                    returned_model_id,
                    raw_text,
                    raw_api_response,
                    request_metadata,
                ) = _request_model(
                    model_name,
                    retry_prompt,
                    image,
                    health_debug_prefix=None,
                    map_file=Path(original_filename).name,
                    perform_health_test=False,
                )
                request_metadata["visible_terms"] = preflight_terms
                _save_llama_raw_debug(
                    timestamp=timestamp,
                    run_id=run_id,
                    file_stem=file_stem,
                    attempt=llama_attempt,
                    raw_text=raw_text,
                    response=raw_api_response,
                    prompt=retry_prompt,
                    max_tokens=MODEL_CONFIGS["Llama"]["max_tokens"],
                    image_file_size=image_debug_path.stat().st_size,
                )
                prompt = retry_prompt
                parsed_before_validation = _load_json_with_repair(raw_text)
                data = parse_model_json(raw_text)
            if (
                model_name == "Llama"
                and len(preflight_terms) >= 5
                and _is_implausible_all_one_result(data)
            ):
                llama_attempt += 1
                calibration_prompt = (
                    f"{prompt}\n\nYou detected visible concept-map content. Regrade using "
                    "that visible evidence. Do not mark criteria as 1 unless truly absent."
                )
                (
                    returned_model_id,
                    raw_text,
                    raw_api_response,
                    request_metadata,
                ) = _request_model(
                    model_name,
                    calibration_prompt,
                    image,
                    health_debug_prefix=None,
                    map_file=Path(original_filename).name,
                    perform_health_test=False,
                )
                request_metadata["visible_terms"] = preflight_terms
                request_metadata["calibration_retry"] = True
                request_metadata["effective_prompt"] = calibration_prompt
                _save_llama_raw_debug(
                    timestamp=timestamp,
                    run_id=run_id,
                    file_stem=file_stem,
                    attempt=llama_attempt,
                    raw_text=raw_text,
                    response=raw_api_response,
                    prompt=calibration_prompt,
                    max_tokens=MODEL_CONFIGS["Llama"]["max_tokens"],
                    image_file_size=image_debug_path.stat().st_size,
                )
                prompt = calibration_prompt
                parsed_before_validation = _load_json_with_repair(raw_text)
                data = parse_model_json(raw_text)
            cleaned_json = _cleaned_json_text(raw_text) if model_name == "Llama" else None
            if model_name == "Llama":
                parsed_after_validation = json.loads(json.dumps(data))
                Path(
                    DEBUG_DIR
                    / (
                        f"{timestamp}_{run_id}_{file_stem}_llama_"
                        "final_parsed_grading.json"
                    )
                ).write_text(
                    json.dumps(parsed_after_validation, indent=2), encoding="utf-8"
                )
                _save_nemotron_trace(
                    timestamp=timestamp,
                    run_id=run_id,
                    file_stem=file_stem,
                    map_filename=Path(original_filename).name,
                    pdf_path=pdf_path,
                    prompt=prompt,
                    image_path=image_debug_path,
                    raw_api_response=raw_api_response,
                    cleaned_json=cleaned_json,
                    parsed_before_validation=parsed_before_validation,
                    parsed_after_validation=parsed_after_validation,
                    request_metadata=request_metadata,
                    error_message=None,
                )
            output_path = OUTPUT_DIR / (
                f"{timestamp}_{run_id}_{file_stem}_{model_name.lower()}.json"
            )
            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            results.append(
                EvaluationResult(model_name, returned_model_id, data, output_path)
            )
        except (ModelResponseError, MalformedResultError) as exc:
            if model_name == "Llama" and image_debug_path is not None:
                debug_path = _save_nemotron_trace(
                    timestamp=timestamp,
                    run_id=run_id,
                    file_stem=file_stem,
                    map_filename=Path(original_filename).name,
                    pdf_path=pdf_path,
                    prompt=prompt,
                    image_path=image_debug_path,
                    raw_api_response=(
                        raw_api_response
                        if raw_api_response is not None
                        else getattr(exc, "raw_response", raw_text)
                    ),
                    cleaned_json=cleaned_json,
                    parsed_before_validation=parsed_before_validation,
                    parsed_after_validation=parsed_after_validation,
                    request_metadata=request_metadata,
                    error_message=str(exc),
                )
            else:
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
