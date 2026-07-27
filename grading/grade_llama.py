"""NVIDIA NIM Llama 3.2 90B Vision single-pass Spring 2025 grader."""

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
MODEL = "meta/llama-3.2-90b-vision-instruct"
PROVIDER = "NVIDIA NIM"
BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY_ENV = "NVIDIA_API_KEY"
MAX_TOKENS = 1800
TEMPERATURE = 0.2
TOP_P = 0.9
TIMEOUT_SECONDS = 120
IMAGE_MIME_TYPE = "image/jpeg"

CATEGORY_FIELDS = {
    "knowledge_acquisition": [
        "basic_science", "health_system_science", "clinical_science",
        "patient_case_information", "determinants_of_health",
    ],
    "integration": [
        "prioritized_differential_diagnosis", "illness_scripts",
        "basic_to_foundational_science", "patient_data_to_clinical_information",
        "patient_data_to_basic_science",
    ],
    "application": ["working_diagnosis_pathophysiology", "patient_data_pathophysiology"],
    "transfer": ["prior_basic_science", "prior_clinical_concepts", "deepens_understanding"],
}
REQUIRED_SCORE_COUNT = sum(len(fields) for fields in CATEGORY_FIELDS.values())


class EmptyLlamaVisionResponseError(RuntimeError):
    def __init__(self, message: str, raw_response: Any, attempts: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.attempts = attempts


class MalformedLlamaVisionJsonError(RuntimeError):
    def __init__(self, attempts: dict[str, Any], message: str | None = None) -> None:
        super().__init__(message or "Llama completed the grading, but its response could not be converted into the required grading JSON.")
        self.attempts = attempts
        self.raw_response = attempts


class NvidiaHttpError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_response = details
        self.status_code = details.get("http_status")
        self.attempts = {"nvidia_http_response": details}


@dataclass
class NvidiaChatCompletion:
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

    def model_dump_json(self, **_: Any) -> str:
        return json.dumps(self.data)


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


def create_nvidia_client() -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("The requests package is not installed.") from exc
    api_key = _secret(API_KEY_ENV)
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")
    return {
        "requests": requests,
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    }


def create_client() -> dict[str, Any]:
    return create_nvidia_client()


def render_pdf_first_page(pdf_path: Path, output_path: Path) -> dict[str, Any]:
    """Render the first page as the production JPEG sent to NVIDIA."""
    import fitz

    with fitz.open(pdf_path) as document:
        if document.page_count < 1:
            raise RuntimeError("The uploaded PDF has no pages.")
        page = document[0]
        max_width_px = 1400
        scale = max_width_px / max(page.rect.width, 1)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False
        )
        image_bytes = pixmap.tobytes("jpeg", jpg_quality=80)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return {
        "path": output_path,
        "base64": base64.b64encode(image_bytes).decode("ascii"),
        "width": pixmap.width,
        "height": pixmap.height,
        "bytes": len(image_bytes),
        "render_matrix": [scale, scale],
        "max_width_px": max_width_px,
        "jpeg_quality": 80,
    }


def _compress_reference_materials(
    materials: list[dict[str, str]] | None, max_characters: int = 4200
) -> str:
    """Keep only compact case, objective, concept, and DDx reference context."""
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
    selected: list[dict[str, str]] = []
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
            selected.append({"filename": filename, "text": text[:remaining]})
            remaining -= len(text[:remaining])
    return format_reference_context(selected)


def _output_contract() -> str:
    lines = []
    for group, fields in CATEGORY_FIELDS.items():
        lines.append(
            f"- {group}: include {', '.join(fields)}. Each criterion needs "
            '"score": <integer from 1 through 4> and "explanation": <one concise sentence>; '
            'also include "overall_decision": "Yes" or "No" and "if_no_explanation".'
        )
    return "\n".join(lines)


def _schema_template() -> dict[str, Any]:
    """Exact application field names with non-numeric placeholders to avoid score anchoring."""
    result: dict[str, Any] = {
        "map_file": "<map filename>", "model": MODEL,
        "overall_meets_expectations": "<Yes or No>",
        "strengths": ["<concise item>"],
        "areas_for_improvement": ["<concise item>"],
        "grading_notes": "<up to two concise sentences>",
    }
    for group, fields in CATEGORY_FIELDS.items():
        result[group] = {
            **{field: {"score": "<integer from 1 through 4>", "explanation": "<one concise evidence-specific sentence>"} for field in fields},
            "overall_decision": "<Yes or No>",
            "if_no_explanation": "<required when overall_decision is No>",
        }
    return result


def build_prompt(
    map_file: str, reference_materials: list[dict[str, str]] | None = None
) -> str:
    reference_context = _compress_reference_materials(reference_materials)
    reference_section = (
        "\nREFERENCE SUMMARY (comparison standard only; not student-map evidence)\n"
        f"{reference_context}\n"
        if reference_context
        else ""
    )
    return (
        "You are grading a medical student concept map using the Spring 2025 Concept Map Feedback "
        "Tool for SUMMATIVE Activities. Inspect all visible concepts, labels, groupings, arrows, and "
        "relationships in the submitted image. Generate every score and written feedback yourself.\n\n"
        + SPRING_2025_RUBRIC
        + reference_section
        + "\nEvaluate only student content visible in the map. Reference material is a comparison "
        "standard only, never student-map evidence. Select the exact rubric score from 1 through 4 "
        "for every criterion. Domain and final decisions must be Yes or No only.\n"
        "Return JSON only using the existing schema. Each explanation is one concise sentence; include "
        "at most 3 strengths and 3 areas_for_improvement; grading_notes is at most 2 sentences. "
        "No markdown, chain-of-thought, or text outside JSON.\n"
        "\nSCORING CALIBRATION\n"
        "Score only visible map evidence. Award 4 only for comprehensive, accurate, clearly integrated, "
        "specifically supported evidence; use 3 for substantial but limited evidence, 2 for partial or "
        "weakly connected evidence, and 1 for absent, largely incorrect, or unsupported evidence. When "
        "between scores, use the lower score. Do not infer missing content from reference material. Each "
        "criterion explanation must cite specific visible concepts, relationships, patient data, diagnoses, "
        "pathways, or omissions.\n\n"
        "OUTPUT CONTRACT\n"
        "Return exactly one valid JSON object. Do not return Markdown, headings, bullets, code fences, "
        "introductory text, trailing commentary, single quotes, comments, trailing commas, NaN, or Infinity. "
        "The first character must be { and the final character must be }. Use double quotes for all keys and "
        "string values. Every required field must be present; do not abbreviate or rename any schema key.\n"
        "REQUIRED JSON SCHEMA (placeholders describe types only; replace them with real values):\n"
        + json.dumps(_schema_template(), separators=(",", ":"))
    )


def _vision_messages(prompt: str, image_base64: str) -> list[dict[str, Any]]:
    """NVIDIA NIM OpenAI-compatible multimodal message: text plus a JPEG data URL."""
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{IMAGE_MIME_TYPE};base64,{image_base64}"}},
    ]}]


def _nvidia_payload(
    messages: list[dict[str, Any]], *, response_format: bool = False, temperature: float = TEMPERATURE
) -> dict[str, Any]:
    payload = {
        "messages": messages,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "top_p": TOP_P,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _post_nvidia(client: dict[str, Any], payload: dict[str, Any]) -> NvidiaChatCompletion:
    endpoint = f"{BASE_URL}/chat/completions"
    started_at = time.monotonic()
    response = client["requests"].post(
        endpoint, headers=client["headers"], json=payload, stream=False, timeout=TIMEOUT_SECONDS
    )
    response_text = response.text
    try:
        data = response.json()
    except (TypeError, ValueError):
        data = None
    headers = dict(getattr(response, "headers", {}) or {})
    details = {
        "provider": "nvidia",
        "http_status": getattr(response, "status_code", None),
        "response_text": response_text,
        "response_json": data,
        "response_headers": headers,
        "request_id_headers": {
            key: value for key, value in headers.items()
            if key.lower() in {"x-request-id", "request-id", "nvcf-request-id", "nvcf-requestid"}
        },
        "elapsed_request_seconds": round(time.monotonic() - started_at, 3),
    }
    if not (200 <= int(getattr(response, "status_code", 0)) < 300):
        body = response_text.strip()
        if isinstance(data, dict):
            body = str(data.get("detail") or data.get("error") or data.get("message") or body)
        raise NvidiaHttpError(f"NVIDIA NIM HTTP {details['http_status']}: {body or 'No error detail returned.'}", details)
    if not isinstance(data, dict):
        raise NvidiaHttpError("NVIDIA NIM returned a non-JSON API response.", details)
    return NvidiaChatCompletion(data=data, http_response=response, transport=details)


def request_grade(
    client: Any, prompt: str, image_base64: str, *, response_format: bool = True
) -> NvidiaChatCompletion:
    return _post_nvidia(
        client, _nvidia_payload(_vision_messages(prompt, image_base64), response_format=response_format)
    )


def request_format_repair(
    client: Any, previous_response: str, *, response_format: bool
) -> NvidiaChatCompletion:
    repair_prompt = (
        "SYSTEM:\nYou are a deterministic JSON formatter. Convert the supplied completed evaluation "
        "into the exact required JSON schema. Preserve grading decisions and evidence. Return JSON only.\n\n"
        "USER:\nThe previous grading response was not valid grading JSON. Do not re-evaluate the concept "
        "map, change scores, add evidence, or alter Yes/No decisions. Preserve explanations as closely as "
        "possible. Return exactly one JSON object, no Markdown or code fences; first character {, final "
        "character }. Every required field must be present.\n\nPREVIOUS RESPONSE:\n"
        + previous_response
        + "\n\nREQUIRED JSON SCHEMA:\n"
        + json.dumps(_schema_template(), separators=(",", ":"))
    )
    return _post_nvidia(
        client,
        _nvidia_payload(
            [{"role": "user", "content": repair_prompt}],
            response_format=response_format,
            temperature=0,
        ),
    )


def _is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return status in {429, 502, 503, 504} or "timeout" in error.__class__.__name__.lower()


def _request_with_retry(request: Any, progress_callback: Any | None = None) -> tuple[Any, dict[str, Any]]:
    try:
        response = request()
        return response, {"request_count": 1, "retry_attempted": False, "http_status": response.transport.get("http_status")}
    except Exception as first_error:
        if not _is_transient(first_error):
            raise
        if progress_callback:
            progress_callback("Llama 3.2 90B Vision request failed transiently. Retrying once...")
        time.sleep(2)
        try:
            response = request()
        except Exception as retry_error:
            retry_error.attempts = {
                "request_count": 2, "retry_attempted": True,
                "first_attempt_error": repr(first_error),
                "retry_attempt_error": repr(retry_error),
                "http_status": getattr(retry_error, "status_code", None),
            }
            raise
        return response, {
            "request_count": 2, "retry_attempted": True,
            "first_attempt_error": repr(first_error),
            "http_status": response.transport.get("http_status"),
        }


def _response_dump(response: Any) -> Any:
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return repr(response)


def _message_value(message: Any, field: str) -> Any:
    return message.get(field) if isinstance(message, dict) else getattr(message, field, None)


def _content_block_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            value = block.get("text") or block.get("content")
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
    return "\n".join(texts)


def _response_diagnostics(response: NvidiaChatCompletion) -> dict[str, Any]:
    data = _response_dump(response)
    choices = response.choices
    first = choices[0] if choices else None
    message = first.get("message", {}) if isinstance(first, dict) else getattr(first, "message", None)
    usage = data.get("usage") if isinstance(data, dict) else None
    return {
        "complete_sanitized_response": data,
        "response_id": data.get("id") if isinstance(data, dict) else None,
        "response_model": data.get("model") if isinstance(data, dict) else None,
        "choices": choices,
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None),
        "message_content": _message_value(message, "content"),
        "message_reasoning": _message_value(message, "reasoning"),
        "message_reasoning_content": _message_value(message, "reasoning_content"),
        "choice_text": first.get("text") if isinstance(first, dict) else getattr(first, "text", None),
        "usage": usage,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
        "http_status": response.transport.get("http_status"),
    }


def response_text(response: NvidiaChatCompletion, attempts: dict[str, Any]) -> str:
    if response is None or not response.choices:
        raise EmptyLlamaVisionResponseError("Llama 3.2 90B Vision returned no response choices.", response, attempts)
    first = response.choices[0]
    message = first.get("message", {}) if isinstance(first, dict) else getattr(first, "message", None)
    content = _message_value(message, "content")
    candidates = [
        content if isinstance(content, str) else None,
        first.get("text") if isinstance(first, dict) else getattr(first, "text", None),
        getattr(response, "output_text", None),
        _content_block_text(content),
    ]
    text = next((item.strip() for item in candidates if isinstance(item, str) and item.strip()), None)
    attempts["response_diagnostics"] = _response_diagnostics(response)
    if text:
        return text
    diagnostics = attempts["response_diagnostics"]
    raise EmptyLlamaVisionResponseError(
        "Llama 3.2 90B Vision returned empty content "
        f"(finish_reason={diagnostics.get('finish_reason')!r}, http_status={diagnostics.get('http_status')!r}).",
        response,
        attempts,
    )


def clean_json_output(text: str) -> str:
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$", "", text).strip()


def _parse_json_object(text: str, attempts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    cleaned = clean_json_output(text)
    decoder = json.JSONDecoder()
    match = re.search(r"\{", cleaned)
    if match:
        try:
            value, end = decoder.raw_decode(cleaned[match.start():])
            # A heading, prose, or trailing commentary is a format contract violation.  Let the
            # single repair call remove it rather than accepting a partly Markdown response.
            if isinstance(value, dict) and not cleaned[:match.start()].strip() and not cleaned[match.start() + end:].strip():
                return json.dumps(value, separators=(",", ":")), value
        except json.JSONDecodeError:
            pass
    error = RuntimeError("Llama 3.2 90B Vision returned malformed grading JSON.")
    error.attempts = attempts
    raise error


def _normalize_scores(result: dict[str, Any]) -> list[dict[str, Any]]:
    normalizations: list[dict[str, Any]] = []
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            continue
        for field in fields:
            item = section.get(field)
            if not isinstance(item, dict) or "score" not in item:
                continue
            original = item["score"]
            normalized: int | None = None
            if isinstance(original, int) and not isinstance(original, bool) and 1 <= original <= 4:
                continue
            if isinstance(original, float) and original.is_integer() and 1 <= original <= 4:
                normalized = int(original)
            elif isinstance(original, str):
                match = re.fullmatch(r"\s*(?:score\s*[:\-]?\s*)?([1-4])(?:\s*/\s*4|\s*[-–—:].*)?\s*", original, re.I)
                if match:
                    normalized = int(match.group(1))
            if normalized is not None:
                item["score"] = normalized
                normalizations.append({"field": f"{group}.{field}.score", "original": original, "normalized": normalized})
    return normalizations


def _validate_existing_schema(result: dict[str, Any]) -> dict[str, Any]:
    """Use the application validator without importing it until a grading call completes."""
    from interface.grading_runner import parse_model_json

    return parse_model_json(json.dumps(result, separators=(",", ":")), normalize_decisions=True)


def _response_format_rejected(error: Exception) -> bool:
    """Identify NVIDIA rejection of the optional OpenAI-compatible JSON-mode field."""
    if not isinstance(error, NvidiaHttpError):
        return False
    details = error.raw_response
    body = " ".join(
        str(value or "") for value in (details.get("response_text"), details.get("response_json"), str(error))
    ).lower()
    return "response_format" in body and any(
        marker in body for marker in ("unsupported", "not support", "unknown", "invalid", "extra", "allowed")
    )


def _score_snapshot(result: dict[str, Any]) -> dict[str, int]:
    """Return only explicitly supplied, valid scores; never synthesize a missing score."""
    scores: dict[str, int] = {}
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            continue
        for field in fields:
            item = section.get(field)
            value = item.get("score") if isinstance(item, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 4:
                scores[f"{group}.{field}"] = value
    return scores


def grade_pdf(
    pdf_path: Path,
    map_file: str,
    debug_prefix: Path,
    reference_materials: list[dict[str, str]] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Grade once, then perform at most one text-only JSON formatting repair."""
    started_at = time.monotonic()
    image_path = Path(f"{debug_prefix}_request.jpg")
    image_info = render_pdf_first_page(pdf_path, image_path)
    image_base64 = image_info["base64"]
    actual_input_path = image_path.parent / "llama32_90b_vision_actual_input.jpg"
    actual_input_path.write_bytes(image_path.read_bytes())
    reference_files = [item["filename"] for item in reference_materials or []]
    prompt = build_prompt(map_file, reference_materials)
    prompt_path = Path(f"{debug_prefix}_prompt.txt")
    prompt_path.write_text(prompt if not reference_files else build_prompt(map_file), encoding="utf-8")
    debug_path = Path(f"{debug_prefix}_debug.json")
    debug: dict[str, Any] = {
        "provider": "nvidia", "model": MODEL, "base_url": BASE_URL,
        "pipeline": "single_pass_multimodal", "request_count": 1,
        "prompt_character_count": len(prompt), "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE, "top_p": TOP_P, "image_mime_type": IMAGE_MIME_TYPE,
        "image_format": "JPEG", "image_width": image_info["width"],
        "image_height": image_info["height"], "image_bytes": image_info["bytes"],
        "image_base64_character_count": len(image_base64), "image_count": 1,
        "image_content_block_included": True, "reference_materials_used": bool(reference_files),
        "reference_files": reference_files, "http_status": None, "finish_reason": None,
        "response_format_requested": True, "response_format_supported": None,
        "initial_http_status": None, "initial_finish_reason": None, "initial_content_length": 0,
        "initial_json_parse_success": False, "initial_schema_validation_success": False,
        "repair_attempted": False, "repair_http_status": None, "repair_finish_reason": None,
        "repair_content_length": 0, "repair_json_parse_success": False,
        "repair_schema_validation_success": False, "image_resent_for_repair": False,
        "final_result_source": "failure",
    }
    debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback("Llama 3.2 90B Vision is grading the concept map...")
    client = create_client()
    try:
        try:
            response, retry_debug = _request_with_retry(
                lambda: request_grade(client, prompt, image_base64, response_format=True), progress_callback
            )
            debug["response_format_supported"] = True
        except Exception as format_error:
            if not _response_format_rejected(format_error):
                raise
            # NVIDIA support differs by served model/version.  Fall back once to the same request
            # without JSON mode; this is not a grading retry and is recorded for deployment diagnosis.
            debug["response_format_supported"] = False
            debug["response_format_rejection"] = getattr(format_error, "raw_response", None)
            response, retry_debug = _request_with_retry(
                lambda: request_grade(client, prompt, image_base64, response_format=False), progress_callback
            )
        attempts = {"first_attempt": _response_dump(response)}
        raw_text = response_text(response, attempts)
    except Exception as exc:
        raw = getattr(exc, "raw_response", None)
        if isinstance(raw, NvidiaChatCompletion):
            debug["complete_response"] = _response_diagnostics(raw)
            debug["http_status"] = raw.transport.get("http_status")
        debug.update({"error": str(exc), "attempts": getattr(exc, "attempts", None), "duration_seconds": round(time.monotonic() - started_at, 3)})
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        raise
    diagnostics = _response_diagnostics(response)
    raw_path = Path(f"{debug_prefix}_grading_raw.txt")
    raw_path.write_text(raw_text, encoding="utf-8")
    debug.update({
        "request_count": retry_debug["request_count"], "retry_attempted": retry_debug["retry_attempted"],
        "initial_http_status": diagnostics["http_status"], "initial_finish_reason": diagnostics["finish_reason"],
        "initial_content_length": len(raw_text), "initial_response": diagnostics,
        "raw_path": str(raw_path),
    })
    initial_parsed: dict[str, Any] | None = None
    initial_scores: dict[str, int] = {}
    initial_error_text = ""
    try:
        _, initial_parsed = _parse_json_object(raw_text, attempts)
        debug["initial_json_parse_success"] = True
        initial_normalizations = _normalize_scores(initial_parsed)
        initial_scores = _score_snapshot(initial_parsed)
        validated = _validate_existing_schema(initial_parsed)
        debug["initial_schema_validation_success"] = True
        parsed_path = Path(f"{debug_prefix}_grading_parsed.json")
        parsed_path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
        debug.update({
            "score_normalizations": initial_normalizations, "parsed_path": str(parsed_path),
            "http_status": diagnostics["http_status"], "finish_reason": diagnostics["finish_reason"],
            "usage": diagnostics["usage"], "response_content_length": len(raw_text),
            "final_result_source": "initial", "duration_seconds": round(time.monotonic() - started_at, 3),
            "payload_shape": {"messages": [{"role": "user", "content": [{"type": "text"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<image-bytes>"}}]}]},
        })
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        return {"model": MODEL, "provider": PROVIDER, "raw_text": raw_text,
                "cleaned_text": json.dumps(validated, separators=(",", ":")), "response": response,
                "prompt": prompt, "prompt_path": prompt_path, "image_path": image_path,
                "raw_path": raw_path, "debug": {**debug, "debug_path": str(debug_path)}}
    except Exception as initial_error:
        initial_error_text = str(initial_error)
        debug["initial_error"] = initial_error_text

    # A complete but malformed/invalid response gets one formatter request only.  It has no image,
    # reference context, or rubric, so it cannot regrade the map.
    debug["repair_attempted"] = True
    if progress_callback:
        progress_callback("Llama completed the grading but returned the wrong format. Converting it to the required JSON structure...")
    try:
        repair_response, repair_retry = _request_with_retry(
            lambda: request_format_repair(
                client, raw_text, response_format=debug.get("response_format_supported") is True
            ),
            progress_callback,
        )
        repair_attempts = {"repair_attempt": _response_dump(repair_response)}
        repair_text = response_text(repair_response, repair_attempts)
        repair_diagnostics = _response_diagnostics(repair_response)
        repair_raw_path = Path(f"{debug_prefix}_format_repair_raw.txt")
        repair_raw_path.write_text(repair_text, encoding="utf-8")
        debug.update({
            "repair_request_count": repair_retry["request_count"],
            "repair_retry_attempted": repair_retry["retry_attempted"],
            "repair_http_status": repair_diagnostics["http_status"],
            "repair_finish_reason": repair_diagnostics["finish_reason"],
            "repair_content_length": len(repair_text), "repair_response": repair_diagnostics,
            "repair_raw_path": str(repair_raw_path),
        })
        _, repaired = _parse_json_object(repair_text, repair_attempts)
        debug["repair_json_parse_success"] = True
        repair_normalizations = _normalize_scores(repaired)
        repaired_scores = _score_snapshot(repaired)
        if initial_parsed is not None and (
            len(initial_scores) != REQUIRED_SCORE_COUNT or repaired_scores != initial_scores
        ):
            raise RuntimeError("The JSON formatter omitted, added, or changed one or more model-generated scores.")
        validated = _validate_existing_schema(repaired)
        debug["repair_schema_validation_success"] = True
        parsed_path = Path(f"{debug_prefix}_grading_parsed.json")
        parsed_path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
        debug.update({
            "repair_score_normalizations": repair_normalizations, "parsed_path": str(parsed_path),
            "final_result_source": "repair", "duration_seconds": round(time.monotonic() - started_at, 3),
        })
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        return {"model": MODEL, "provider": PROVIDER, "raw_text": repair_text,
                "cleaned_text": json.dumps(validated, separators=(",", ":")), "response": repair_response,
                "prompt": prompt, "prompt_path": prompt_path, "image_path": image_path,
                "raw_path": repair_raw_path, "debug": {**debug, "debug_path": str(debug_path)}}
    except Exception as repair_error:
        debug.update({
            "repair_error": str(repair_error), "repair_attempts": getattr(repair_error, "attempts", None),
            "duration_seconds": round(time.monotonic() - started_at, 3), "final_result_source": "failure",
        })
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        schema_error = debug["initial_json_parse_success"] and not debug["initial_schema_validation_success"]
        message = (
            "Llama returned JSON, but required grading fields were missing or invalid."
            if schema_error else
            "Llama completed the grading, but its response could not be converted into the required grading JSON."
        )
        raise MalformedLlamaVisionJsonError(debug, message) from repair_error
