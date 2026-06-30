"""Upload-driven grading pipeline used by the Streamlit demo.

The command-line graders remain independent. This module mirrors their proven
PDF rendering flow while accepting any uploaded PDF.
"""

from __future__ import annotations

import base64
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
NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

GRADER_MODULES = {
    "Gemma": None,
    "Nemotron": None,
}

MODEL_SELECTION_ALIASES: dict[str, str] = {}

MODEL_CONFIGS = {
    "Gemma": {
        "model_id": GEMMA_MODEL,
        "max_tokens": 2000,
    },
    "Nemotron": {
        "model_id": NEMOTRON_MODEL,
        "max_tokens": 2000,
    },
}

MODEL_PROVIDER_INFO = {
    "Gemma": {
        "provider": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": GEMMA_MODEL,
    },
    "Nemotron": {
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
    if selection == "Both":
        return ["Gemma", "Nemotron"]
    selection = MODEL_SELECTION_ALIASES.get(selection, selection)
    if selection not in MODEL_CONFIGS:
        raise GradingError(f"Unknown model selection: {selection}")
    return [selection]


def model_debug_lines(model_names: Iterable[str] | None = None) -> list[str]:
    """Return provider/model debug lines safe for display or logs."""
    if model_names is None:
        return [
            "Gemma provider: OpenRouter",
            f"Gemma model: {GEMMA_MODEL}",
            "Nemotron provider: NVIDIA NIM",
            f"Nemotron model: {NEMOTRON_MODEL}",
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
    """Render the uploaded PDF as a deployment-safe base64 PNG."""
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
            # Lower resolution for hosted model token compatibility
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            image_bytes = pixmap.tobytes("png")
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
    """Build the same full Spring 2025 prompt for either model."""
    _ = model_name
    return build_web_prompt(map_file, model_id)


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


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Extract, parse, repair when possible, and validate grading JSON."""
    if not raw_text or not raw_text.strip():
        raise ModelResponseError("The model returned an empty response.")

    if "{" not in raw_text:
        raise MalformedResultError("The model response did not contain a JSON object.")

    result = _load_json_with_repair(raw_text)
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
    if model_name == "Nemotron":
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


def _request_model(
    model_name: str,
    prompt: str,
    image: str,
    max_tokens: int | None = None,
    request_timeout: float | None = None,
) -> tuple[str, str]:
    config = MODEL_CONFIGS[model_name]
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
        },
    ]

    try:
        client = _create_client(
            model_name, disable_sdk_retries=request_timeout is not None
        )
        if model_name == "Nemotron":
            for line in model_debug_lines([model_name]):
                print(line)
        request_options: dict[str, Any] = {
            "model": config["model_id"],
            "max_tokens": max_tokens or config["max_tokens"],
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        if request_timeout is not None:
            request_options["timeout"] = request_timeout
        response = client.chat.completions.create(
            **request_options
        )
    except Exception as exc:
        message = str(exc)
        if _is_input_limit_error(message):
            message = (
                "Input is too large for the current model limit. "
                "Try a smaller PDF/image or use the local CLI pipeline."
            )
        provider = "NVIDIA NIM" if model_name == "Nemotron" else "OpenRouter"
        raise ModelResponseError(
            f"{model_name} {provider} API request failed: {message}",
            raw_response=repr(exc),
        ) from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise ModelResponseError(
            f"{model_name} returned no response choices.",
            raw_response=response,
        )
    try:
        text = choices[0].message.content
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
    return config["model_id"], text


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
        try:
            prompt = build_model_prompt(
                model_name, Path(original_filename).name, model_id
            )
            returned_model_id, raw_text = _request_model(model_name, prompt, image)
            data = parse_model_json(raw_text)
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
