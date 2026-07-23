"""Groq Qwen 3.6 27B two-stage grader for Spring 2025 evaluation."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grading.spring_2025_prompt import SPRING_2025_RUBRIC
from interface.reference_materials import format_reference_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen/qwen3.6-27b"
PROVIDER = "Groq"
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_ENV = "GROQ_API_KEY"
MAX_TOKENS = 1600
EXTRACTION_MAX_TOKENS = 900
TIMEOUT_SECONDS = 180
IMAGE_MIME_TYPE = "image/jpeg"
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


class EmptyLlamaVisionResponseError(RuntimeError):
    """Qwen 3.6 27B returned no usable completion content."""

    def __init__(self, message: str, raw_response: Any, attempts: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.attempts = attempts


class MalformedLlamaVisionJsonError(RuntimeError):
    def __init__(self, attempts: dict[str, Any]) -> None:
        super().__init__("Qwen 3.6 27B returned malformed JSON after one repair attempt.")
        self.attempts = attempts


class GroqQwenHttpError(RuntimeError):
    """Groq returned an HTTP response that must remain visible to the user."""

    def __init__(self, message: str, response_details: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_response = response_details
        self.status_code = response_details.get("http_status")
        self.attempts = {"groq_http_response": response_details}


@dataclass
class GroqChatCompletion:
    """Small adapter preserving the response interface used by this module."""

    data: dict[str, Any]
    http_response: Any
    transport: dict[str, Any]

    @property
    def choices(self) -> list[Any]:
        return self.data.get("choices") or []

    @property
    def output_text(self) -> Any:
        return self.data.get("output_text")

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self.data


def _secret(name: str) -> str | None:
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


def create_client() -> Any:
    return create_groq_client()


def create_groq_client() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "The requests package is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    api_key = _secret(API_KEY_ENV)
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return {
        "requests": requests,
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    }


def _is_retryable_transport_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
    return (isinstance(status_code, int) and 500 <= status_code <= 599) or "timeout" in error.__class__.__name__.lower()


def _groq_rate_limit_delay(error: Exception) -> float | None:
    """Return Groq's requested retry delay only for rate_limit_exceeded."""
    if getattr(error, "status_code", None) != 429:
        return None
    details = getattr(error, "raw_response", {})
    body = str(details.get("response_text", "")) if isinstance(details, dict) else ""
    parsed = details.get("response_json") if isinstance(details, dict) else None
    error_data = parsed.get("error") if isinstance(parsed, dict) else None
    code = error_data.get("code") if isinstance(error_data, dict) else None
    if code != "rate_limit_exceeded" and "rate_limit_exceeded" not in body:
        return None
    match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", body, re.IGNORECASE)
    return float(match.group(1)) if match else 20.0


def _request_with_retry(
    request: Any,
    stage_name: str = "request",
    progress_callback: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    started_at = time.monotonic()
    try:
        response = request()
        return response, {"attempt_number": 1, "http_status": getattr(getattr(response, "http_response", None), "status_code", 200), "request_duration_seconds": round(time.monotonic() - started_at, 3), "retry_attempted": False}
    except Exception as first_error:
        rate_limit_delay = _groq_rate_limit_delay(first_error)
        if rate_limit_delay is None and not _is_retryable_transport_error(first_error):
            raise
        retry_delay = rate_limit_delay + 2 if rate_limit_delay is not None else 5
        if rate_limit_delay is not None and progress_callback:
            progress_callback(
                "Qwen rate limit reached. Waiting approximately "
                f"{round(retry_delay)} seconds before continuing..."
            )
        time.sleep(retry_delay)
        if rate_limit_delay is not None and progress_callback:
            progress_callback("Retrying Qwen grading...")
        retry_started_at = time.monotonic()
        try:
            response = request()
        except Exception as retry_error:
            setattr(retry_error, "attempts", {
                "attempt_number": 2,
                "first_attempt_error": repr(first_error),
                "first_attempt_response": getattr(first_error, "attempts", None),
                "retry_attempt_error": repr(retry_error),
                "retry_attempt_response": getattr(retry_error, "attempts", None),
                "http_status": getattr(retry_error, "status_code", None),
                "retry_attempted": True,
                "stage": stage_name,
                "retry_reason": "rate_limit_exceeded" if rate_limit_delay is not None else "transport_error",
                "retry_delay_seconds": retry_delay,
            })
            raise
        return response, {
            "attempt_number": 2,
            "http_status": getattr(getattr(response, "http_response", None), "status_code", 200),
            "request_duration_seconds": round(time.monotonic() - retry_started_at, 3),
            "retry_attempted": True,
            "first_attempt_error": repr(first_error),
            "first_attempt_response": getattr(first_error, "attempts", None),
            "stage": stage_name,
            "retry_reason": "rate_limit_exceeded" if rate_limit_delay is not None else "transport_error",
            "retry_delay_seconds": retry_delay,
        }


def render_pdf_first_page(pdf_path: Path, output_path: Path) -> dict[str, Any]:
    """Render first PDF page to a small compressed JPEG."""
    import fitz

    with fitz.open(pdf_path) as document:
        if document.page_count < 1:
            raise RuntimeError("The uploaded PDF has no pages.")
        page = document[0]
        max_width_px = 1400
        scale = max_width_px / max(page.rect.width, 1)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image_bytes = pixmap.tobytes("jpeg", jpg_quality=80)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return {
        "base64": base64.b64encode(image_bytes).decode("utf-8"),
        "path": output_path,
        "width": pixmap.width,
        "height": pixmap.height,
        "bytes": len(image_bytes),
        "render_matrix": f"fitz.Matrix({scale:.4f}, {scale:.4f})",
        "max_width_px": max_width_px,
        "jpeg_quality": 80,
    }


def schema(map_file: str) -> dict[str, Any]:
    result: dict[str, Any] = {"map_file": map_file, "model": MODEL}
    for group, fields in CATEGORY_FIELDS.items():
        result[group] = {
            field: {"score": 1, "explanation": ""}
            for field in fields
        }
        result[group]["overall_decision"] = "No"
        result[group]["if_no_explanation"] = ""
    result["overall_meets_expectations"] = "No"
    result["strengths"] = ["", ""]
    result["areas_for_improvement"] = ["", ""]
    result["grading_notes"] = ""
    return result


def build_prompt(
    map_file: str, reference_materials: list[dict[str, str]] | None = None
) -> str:
    return build_stage_two_prompt(map_file, {}, reference_materials)


EXTRACTION_FIELDS = (
    "main_topic",
    "patient_data",
    "basic_science_concepts",
    "clinical_science_concepts",
    "health_system_science_concepts",
    "determinants_of_health",
    "differential_diagnoses",
    "relationships",
    "pathophysiology_flows",
    "prior_or_transfer_knowledge",
    "unclear_or_unreadable_content",
)


def extraction_schema() -> dict[str, Any]:
    """The non-grading, visible-content-only Stage 1 response shape."""
    return {
        "main_topic": "",
        "patient_data": [],
        "basic_science_concepts": [],
        "clinical_science_concepts": [],
        "health_system_science_concepts": [],
        "determinants_of_health": [],
        "differential_diagnoses": [],
        "relationships": [{"from": "", "to": "", "relationship": ""}],
        "pathophysiology_flows": [],
        "prior_or_transfer_knowledge": [],
        "unclear_or_unreadable_content": [],
    }


def build_extraction_prompt() -> str:
    return (
        "Extract only visibly present content from this medical concept map. Do not grade it. "
        "Do not infer missing content. Preserve specific patient facts when readable and "
        "write every visible arrow or relationship explicitly. Use concise phrases, not paragraphs. "
        "Limit patient_data to 12 items; basic_science_concepts and clinical_science_concepts to 15 "
        "each; health_system_science_concepts, determinants_of_health, differential_diagnoses, and "
        "pathophysiology_flows to 8 each; relationships to 15; and prior_or_transfer_knowledge to 10. "
        "Do not omit important visible evidence solely to meet these limits. Return only valid JSON with "
        "this exact structure:\n"
        + json.dumps(extraction_schema(), separators=(",", ":"))
    )


def _compress_reference_materials(
    materials: list[dict[str, str]] | None, max_characters: int = 4200
) -> str:
    """Keep only case, objective, unit-concept, and DDx context for Stage 2."""
    selected: list[dict[str, str]] = []
    keywords = re.compile(
        r"patient|case|history|chief|symptom|finding|diagnos|differential|ddx|"
        r"objective|outcome|learn|pathophys|physiology|anatom|histolog|biochem|"
        r"genetic|pharmacol|clinical|health system|determinant|social",
        re.IGNORECASE,
    )
    discard = re.compile(
        r"copyright|all rights reserved|poll|clicker|audience response|slide \d+|"
        r"www\.|http[s]?://|page \d+ of \d+",
        re.IGNORECASE,
    )
    remaining = max_characters
    for material in materials or []:
        filename = str(material.get("filename", "")).strip()
        seen: set[str] = set()
        kept: list[str] = []
        for raw_line in str(material.get("text", "")).splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            key = line.casefold()
            if not line or key in seen or discard.search(line) or not keywords.search(line):
                continue
            seen.add(key)
            kept.append(line[:350])
        text = "\n".join(kept)
        if filename and text and remaining > 0:
            clipped = text[:remaining]
            selected.append({"filename": filename, "text": clipped})
            remaining -= len(clipped)
    return format_reference_context(selected)


def _output_contract() -> str:
    domain_lines = []
    for group, fields in CATEGORY_FIELDS.items():
        domain_lines.append(
            f"- {group}: criterion objects for {', '.join(fields)}; each criterion has "
            "score (required integer 1-4) and explanation (brief string); also "
            "overall_decision (Yes/No) and if_no_explanation (string)."
        )
    return "\n".join(domain_lines)


def build_stage_two_prompt(
    map_file: str,
    extracted_content: dict[str, Any],
    reference_materials: list[dict[str, str]] | None,
) -> str:
    reference_context = _compress_reference_materials(reference_materials)
    reference_section = (
        "\nREFERENCE SUMMARY (comparison standard only; not student-map evidence)\n"
        + reference_context
        + "\n"
        if reference_context
        else ""
    )
    return (
        "You are grading a medical student concept map using the Spring 2025 Concept Map Feedback "
        "Tool for SUMMATIVE Activities. Use this exact rubric as the sole scoring authority.\n\n"
        + SPRING_2025_RUBRIC
        + reference_section
        + "\nEXTRACTED STUDENT CONCEPT MAP CONTENT (the only student-map evidence)\n"
        + json.dumps(extracted_content, separators=(",", ":"))
        + "\n\nGrade only the extracted content. For each criterion, select the exact 1-4 descriptor "
        "that best matches it; do not use hidden thresholds or averages. Domain and final Yes/No "
        "decisions answer the rubric questions holistically. Keep each criterion explanation to one "
        "short sentence. Do not include chain-of-thought or extended reasoning.\n"
        "Return only valid JSON. Required fields: map_file, model, knowledge_acquisition, integration, "
        "application, transfer, overall_meets_expectations (Yes/No), strengths (list), "
        "areas_for_improvement (list), and grading_notes (string).\n"
        + _output_contract()
    )


def _groq_payload(
    messages: list[dict[str, Any]], max_completion_tokens: int = MAX_TOKENS
) -> dict[str, Any]:
    """Use Groq's Qwen chat-completions request fields without JSON mode."""
    return {
        "messages": messages,
        "model": MODEL,
        "max_completion_tokens": max_completion_tokens,
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "reasoning_format": "hidden",
    }


def _post_groq(client: dict[str, Any], payload: dict[str, Any]) -> GroqChatCompletion:
    endpoint = f"{BASE_URL}/chat/completions"
    started_at = time.monotonic()
    response = client["requests"].post(
        endpoint,
        headers=client["headers"],
        json=payload,
        stream=False,
        timeout=TIMEOUT_SECONDS,
    )
    response_text = response.text
    try:
        data = response.json()
    except (ValueError, TypeError):
        data = None

    headers = dict(getattr(response, "headers", {}) or {})
    request_headers = {
        key: value
        for key, value in headers.items()
        if key.lower() in {"x-request-id", "request-id", "x-correlation-id", "nvcf-request-id", "nvcf-requestid"}
    }
    response_details = {
        "http_status": getattr(response, "status_code", None),
        "response_text": response_text,
        "response_json": data,
        "request_id_headers": request_headers,
        "response_headers": headers,
        "elapsed_request_seconds": round(time.monotonic() - started_at, 3),
    }

    if not (200 <= int(getattr(response, "status_code", 0)) < 300):
        body_detail = response_text.strip()
        if isinstance(data, dict):
            body_detail = str(data.get("detail") or data.get("error") or data.get("message") or body_detail)
        message = f"Groq HTTP {response_details['http_status']}: {body_detail or 'No error detail returned.'}"
        raise GroqQwenHttpError(message, response_details)

    if not isinstance(data, dict):
        raise GroqQwenHttpError("Groq returned a non-JSON API response.", response_details)
    return GroqChatCompletion(data=data, http_response=response, transport=response_details)


def _vision_messages(prompt: str, image_base64: str) -> list[dict[str, Any]]:
    # Groq's Qwen vision endpoint accepts OpenAI-compatible image_url content.
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{IMAGE_MIME_TYPE};base64,{image_base64}"
                    },
                },
            ],
        }
    ]


def request_extraction(client: Any, image_base64: str) -> Any:
    """Stage 1: extract only visible map content from the production JPEG."""
    return _post_groq(
        client,
        _groq_payload(
            _vision_messages(build_extraction_prompt(), image_base64),
            max_completion_tokens=EXTRACTION_MAX_TOKENS,
        ),
    )


def request_grade(client: Any, prompt: str) -> Any:
    """Stage 2: grade the Stage 1 JSON without resending the concept-map image."""
    return _post_groq(client, _groq_payload([{"role": "user", "content": prompt}]))


def _response_debug_value(response: Any) -> Any:
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            pass
    return repr(response)


def _response_shape(response: Any) -> dict[str, Any]:
    choices = getattr(response, "choices", None)
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message", {}) if isinstance(first, dict) else getattr(first, "message", None)
    response_dump = _response_debug_value(response)
    return {
        "http_status": getattr(getattr(response, "http_response", None), "status_code", 200),
        "response_headers": dict(getattr(getattr(response, "http_response", None), "headers", {}) or {}),
        "top_level_keys": list(response_dump.keys()) if isinstance(response_dump, dict) else [],
        "choices_length": len(choices) if isinstance(choices, list) else 0,
        "message_content": message.get("content") if isinstance(message, dict) else getattr(message, "content", None),
        "choice_text": first.get("text") if isinstance(first, dict) else getattr(first, "text", None),
        "reasoning_content": message.get("reasoning_content") if isinstance(message, dict) else getattr(message, "reasoning_content", None),
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None),
    }


def response_text(response: Any, attempts: dict[str, Any]) -> str:
    if response is None:
        raise EmptyLlamaVisionResponseError("Qwen 3.6 27B returned no response.", response, attempts)
    choices = getattr(response, "choices", None)
    if not choices:
        raise EmptyLlamaVisionResponseError("Qwen 3.6 27B returned no response choices.", response, attempts)
    first = choices[0]
    message = first.get("message", {}) if isinstance(first, dict) else getattr(first, "message", None)
    candidates = [
        message.get("content") if isinstance(message, dict) else getattr(message, "content", None),
        first.get("text") if isinstance(first, dict) else getattr(first, "text", None),
        message.get("reasoning_content") if isinstance(message, dict) else getattr(message, "reasoning_content", None),
        getattr(response, "output_text", None),
    ]
    text = next((value for value in candidates if isinstance(value, str) and value.strip()), None)
    if text is None:
        raise EmptyLlamaVisionResponseError("Qwen 3.6 27B returned empty content.", response, attempts)
    return text


def request_json_repair(client: Any, malformed_output: str, map_file: str) -> Any:
    repair_prompt = (
        "Return the same evaluation as valid JSON only. Do not regrade or change scores.\n"
        "Required fields: map_file, model, knowledge_acquisition, integration, application, transfer, "
        "overall_meets_expectations, strengths, areas_for_improvement, grading_notes.\n"
        + _output_contract()
        + "\nMalformed output:\n"
        + malformed_output
    )
    return _post_groq(client, _groq_payload([{"role": "user", "content": repair_prompt}]))


def request_extraction_repair(client: Any, malformed_output: str) -> Any:
    repair_prompt = (
        "Return the same extracted concept-map content as valid JSON only. Do not grade or infer. "
        "Required extraction structure:\n"
        + json.dumps(extraction_schema(), separators=(",", ":"))
        + "\nMalformed output:\n"
        + malformed_output
    )
    return _post_groq(client, _groq_payload([{"role": "user", "content": repair_prompt}]))


def clean_json_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0).strip() if match else text


def _parse_json_object(text: str, error_message: str, attempts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    cleaned = clean_json_output(text)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        error = RuntimeError(error_message)
        error.attempts = {**attempts, "json_error": str(exc)}
        raise error from exc
    if not isinstance(value, dict):
        error = RuntimeError(error_message)
        error.attempts = {**attempts, "json_error": "JSON root must be an object."}
        raise error
    return cleaned, value


def _validate_extraction(value: dict[str, Any], attempts: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in EXTRACTION_FIELDS if field not in value]
    invalid_lists = [
        field for field in EXTRACTION_FIELDS
        if field != "main_topic" and field in value and not isinstance(value[field], list)
    ]
    relationships_valid = all(
        isinstance(item, dict) and all(isinstance(item.get(key, ""), str) for key in ("from", "to", "relationship"))
        for item in value.get("relationships", [])
    )
    if missing or invalid_lists or not isinstance(value.get("main_topic"), str) or not relationships_valid:
        error = RuntimeError("Qwen 3.6 27B returned an invalid extraction response.")
        error.attempts = {
            **attempts,
            "missing_extraction_fields": missing,
            "invalid_extraction_list_fields": invalid_lists,
            "relationships_valid": relationships_valid,
        }
        raise error
    return value


def _normalize_qwen_scores(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert only unambiguous 1–4 score representations before validation."""
    normalizations: list[dict[str, Any]] = []

    def as_score(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value if 1 <= value <= 4 else None
        if isinstance(value, float):
            if value.is_integer() and 1 <= int(value) <= 4:
                return int(value)
            return None
        if not isinstance(value, str):
            return None

        text = value.strip()
        if re.fullmatch(r"[1-4]", text):
            return int(text)
        score_match = re.fullmatch(r"score\s*[:\-]?\s*([1-4])", text, re.IGNORECASE)
        if score_match:
            return int(score_match.group(1))
        fraction_match = re.fullmatch(r"([1-4])\s*/\s*4", text)
        if fraction_match:
            return int(fraction_match.group(1))
        descriptor_match = re.fullmatch(r"([1-4])\s*[-–—:]\s*\D.+", text)
        if descriptor_match:
            return int(descriptor_match.group(1))
        return None

    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            continue
        for field in fields:
            criterion = section.get(field)
            if not isinstance(criterion, dict) or "score" not in criterion:
                continue
            original = criterion["score"]
            normalized = as_score(original)
            if normalized is None or (isinstance(original, int) and not isinstance(original, bool)):
                continue
            criterion["score"] = normalized
            normalizations.append(
                {
                    "field": f"{group}.{field}.score",
                    "original": original,
                    "normalized": normalized,
                }
            )
    return normalizations


def _read_response_with_one_empty_retry(
    request: Any, stage_name: str, progress_callback: Any | None = None
) -> tuple[str, Any, dict[str, Any], dict[str, Any]]:
    """Bound each stage to at most two calls, including empty-choice retries."""
    response, transport_debug = _request_with_retry(
        request, stage_name=stage_name, progress_callback=progress_callback
    )
    attempts: dict[str, Any] = {"first_attempt": _response_debug_value(response)}
    try:
        return response_text(response, attempts), response, transport_debug, attempts
    except EmptyLlamaVisionResponseError as first_error:
        if transport_debug.get("retry_attempted"):
            raise first_error
        time.sleep(5)
        retry_started = time.monotonic()
        try:
            retry_response = request()
        except Exception as retry_error:
            retry_error.attempts = {
                "stage": stage_name,
                "first_attempt": attempts["first_attempt"],
                "retry_attempt_error": repr(retry_error),
                "retry_attempt_response": getattr(retry_error, "attempts", None),
            }
            raise
        attempts["retry_attempt"] = _response_debug_value(retry_response)
        transport_debug.update(
            {
                "empty_response_retry_attempted": True,
                "empty_response_retry_duration_seconds": round(time.monotonic() - retry_started, 3),
            }
        )
        try:
            return response_text(retry_response, attempts), retry_response, transport_debug, attempts
        except EmptyLlamaVisionResponseError as retry_error:
            raise EmptyLlamaVisionResponseError(str(retry_error), retry_response, attempts) from first_error


def _vision_diagnostic_enabled() -> bool:
    return os.getenv("QWEN36_VISION_DIAGNOSTIC", "").strip() == "1"


def request_vision_diagnostic(client: Any, image_base64: str) -> Any:
    diagnostic_prompt = (
        "Read this concept map carefully.\n\n"
        "Return plain text only.\n\n"
        "1. What is the main medical topic or diagnosis?\n"
        "2. List up to 20 specific medical concepts or phrases you can clearly read.\n"
        "3. List any patient-specific information you can read.\n"
        "4. Describe at least 5 visible relationships or arrows between concepts.\n"
        "5. State whether the image text is:\n"
        "   - Clearly readable\n"
        "   - Partially readable\n"
        "   - Mostly unreadable"
    )
    return _post_groq(client, _groq_payload(_vision_messages(diagnostic_prompt, image_base64)))


def grade_pdf(
    pdf_path: Path,
    map_file: str,
    debug_prefix: Path,
    reference_materials: list[dict[str, str]] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    image_path = Path(f"{debug_prefix}_request.jpg")
    image_info = render_pdf_first_page(pdf_path, image_path)
    image_base64 = str(image_info["base64"])
    actual_input_path = image_path.parent / "qwen36_actual_input.jpg"
    actual_input_path.write_bytes(image_path.read_bytes())
    diagnostic_enabled = _vision_diagnostic_enabled()
    if diagnostic_enabled:
        client = create_client()
        response, transport_debug = _request_with_retry(
            lambda: request_vision_diagnostic(client, image_base64)
        )
        raw_text = response_text(response, {"diagnostic_attempt": _response_debug_value(response)})
        diagnostic_path = image_path.parent / "qwen36_vision_diagnostic.txt"
        diagnostic_path.write_text(raw_text, encoding="utf-8")
        return {
            "model": MODEL,
            "provider": PROVIDER,
            "raw_text": raw_text,
            "response": response,
            "diagnostic": True,
            "debug": {
                "provider": PROVIDER,
                "base_url": BASE_URL,
                "model": MODEL,
                "image_path": str(image_path),
                "actual_input_path": str(actual_input_path),
                "image_mime_type": IMAGE_MIME_TYPE,
                "image_width": image_info["width"],
                "image_height": image_info["height"],
                "image_bytes": image_info["bytes"],
                "render_matrix": image_info["render_matrix"],
                "jpeg_quality": image_info["jpeg_quality"],
                "diagnostic_path": str(diagnostic_path),
                "payload_shape": {"messages": [{"role": "user", "content": [{"type": "text"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<image-bytes>"}}]}], "stream": False, "reasoning_format": "hidden"},
                "raw_response": _response_debug_value(response),
                "groq_http_response": response.transport,
                **transport_debug,
            },
        }
    reference_files = [item["filename"] for item in reference_materials or []]
    debug_path = Path(f"{debug_prefix}_debug.json")
    debug_payload = {
        "provider": PROVIDER,
        "base_url": BASE_URL,
        "model": MODEL,
        "image_path": str(image_info["path"]),
        "actual_input_path": str(actual_input_path),
        "image_mime_type": IMAGE_MIME_TYPE,
        "image_width": image_info["width"],
        "image_height": image_info["height"],
        "image_bytes": image_info["bytes"],
        "render_matrix": image_info["render_matrix"],
        "max_width_px": image_info["max_width_px"],
        "jpeg_quality": image_info["jpeg_quality"],
        "reference_materials_used": bool(reference_files),
        "reference_files": reference_files,
        "max_tokens": MAX_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "pipeline": "stage_1_image_extraction_then_stage_2_text_grading",
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    client = create_client()
    if progress_callback:
        progress_callback("Extracting Qwen evidence...")
    extraction_started = time.monotonic()
    extraction_raw, extraction_response, extraction_transport, extraction_attempts = (
        _read_response_with_one_empty_retry(
            lambda: request_extraction(client, image_base64), "extraction", progress_callback
        )
    )
    extraction_raw_path = Path(f"{debug_prefix}_extraction_raw.txt")
    extraction_raw_path.write_text(extraction_raw, encoding="utf-8")
    debug_payload["stage_1_extraction"] = {
        "duration_seconds": round(time.monotonic() - extraction_started, 3),
        "raw_path": str(extraction_raw_path),
        "raw_response": _response_debug_value(extraction_response),
        "groq_http_response": extraction_response.transport,
        "response_shape": _response_shape(extraction_response),
        "attempts": extraction_attempts,
        "transport": extraction_transport,
        "max_completion_tokens": EXTRACTION_MAX_TOKENS,
        "payload_shape": {"messages": [{"role": "user", "content": [{"type": "text"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<image-bytes>"}}]}]},
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    try:
        _, extracted_content = _parse_json_object(
            extraction_raw,
            "Qwen 3.6 27B extraction response was not valid JSON.",
            extraction_attempts,
        )
        extracted_content = _validate_extraction(extracted_content, extraction_attempts)
    except RuntimeError:
        extraction_repair = request_extraction_repair(client, extraction_raw)
        extraction_repair_text = response_text(extraction_repair, extraction_attempts)
        extraction_attempts["repair_attempt"] = extraction_repair_text
        extraction_raw_path.write_text(
            extraction_raw + "\n\n--- repair_attempt ---\n" + extraction_repair_text,
            encoding="utf-8",
        )
        _, extracted_content = _parse_json_object(
            extraction_repair_text,
            "Qwen 3.6 27B extraction response was not valid JSON after repair.",
            extraction_attempts,
        )
        extracted_content = _validate_extraction(extracted_content, extraction_attempts)
    extraction_parsed_path = Path(f"{debug_prefix}_extraction_parsed.json")
    extraction_parsed_path.write_text(json.dumps(extracted_content, indent=2), encoding="utf-8")
    debug_payload["stage_1_extraction"].update({
        "duration_seconds": round(time.monotonic() - extraction_started, 3),
        "parsed_path": str(extraction_parsed_path),
        "repair_attempt": extraction_attempts.get("repair_attempt"),
    })
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    prompt = build_stage_two_prompt(map_file, extracted_content, reference_materials)
    prompt_path = Path(f"{debug_prefix}_prompt.txt")
    if reference_files:
        prompt_path.write_text(
            "Reference text omitted from debug output. Files used: "
            + ", ".join(reference_files)
            + "\n\n"
            + build_stage_two_prompt(map_file, extracted_content, None),
            encoding="utf-8",
        )
    else:
        prompt_path.write_text(prompt, encoding="utf-8")

    if progress_callback:
        progress_callback("Grading extracted evidence with Qwen...")
    grading_started = time.monotonic()
    grading_raw, response, grading_transport, attempts = _read_response_with_one_empty_retry(
        lambda: request_grade(client, prompt), "grading", progress_callback
    )
    grading_raw_path = Path(f"{debug_prefix}_grading_raw.txt")
    grading_raw_path.write_text(grading_raw, encoding="utf-8")
    debug_payload["stage_2_grading"] = {
        "duration_seconds": round(time.monotonic() - grading_started, 3),
        "prompt_path": str(prompt_path),
        "prompt_characters": len(prompt),
        "max_completion_tokens": MAX_TOKENS,
        "raw_path": str(grading_raw_path),
        "raw_response": _response_debug_value(response),
        "groq_http_response": response.transport,
        "response_shape": _response_shape(response),
        "attempts": attempts,
        "transport": grading_transport,
        "stage_1_reused": True,
        "payload_shape": {"messages": [{"role": "user", "content": "<rubric + extracted content>"}]},
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    raw_text = grading_raw
    try:
        cleaned_text, parsed_grading = _parse_json_object(
            raw_text,
            "Qwen 3.6 27B returned malformed grading JSON.",
            attempts,
        )
    except RuntimeError:
        repair_response = request_json_repair(client, raw_text, map_file)
        repair_text = response_text(repair_response, attempts)
        attempts["repair_attempt"] = repair_text
        grading_raw_path.write_text(
            grading_raw + "\n\n--- repair_attempt ---\n" + repair_text,
            encoding="utf-8",
        )
        raw_text = repair_text
        try:
            cleaned_text, parsed_grading = _parse_json_object(
                raw_text,
                "Qwen 3.6 27B returned malformed grading JSON after one repair attempt.",
                attempts,
            )
        except RuntimeError as exc:
            raise MalformedLlamaVisionJsonError(attempts) from exc
    grading_parsed_path = Path(f"{debug_prefix}_grading_parsed.json")
    grading_parsed_path.write_text(json.dumps(parsed_grading, indent=2), encoding="utf-8")
    def _parsed_section(group: str) -> dict[str, Any]:
        section = parsed_grading.get(group)
        return section if isinstance(section, dict) else {}

    pre_normalization_scores = {
        group: {
            field: _parsed_section(group).get(field, {}).get("score")
            if isinstance(_parsed_section(group).get(field), dict)
            else None
            for field in fields
        }
        for group, fields in CATEGORY_FIELDS.items()
    }
    pre_normalization_decisions = {
        group: _parsed_section(group).get("overall_decision")
        for group in CATEGORY_FIELDS
    }
    score_normalizations = _normalize_qwen_scores(parsed_grading)
    # The runner validates this normalized serialization; raw and parsed debug
    # artifacts above retain the model's original response for auditability.
    cleaned_text = json.dumps(parsed_grading, separators=(",", ":"))
    debug_payload["stage_2_grading"].update({
        "duration_seconds": round(time.monotonic() - grading_started, 3),
        "parsed_path": str(grading_parsed_path),
        "repair_attempt": attempts.get("repair_attempt"),
        "pre_normalization_criterion_scores": pre_normalization_scores,
        "pre_normalization_domain_decisions": pre_normalization_decisions,
        "pre_normalization_overall_meets_expectations": parsed_grading.get(
            "overall_meets_expectations"
        ),
        "score_normalizations": score_normalizations,
    })
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    return {
        "model": MODEL,
        "provider": PROVIDER,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "response": response,
        "prompt": prompt,
        "prompt_path": prompt_path,
        "image_path": image_path,
        "raw_path": grading_raw_path,
        "debug": {
            **debug_payload,
            "debug_path": str(debug_path),
            "raw_path": str(grading_raw_path),
        },
    }
