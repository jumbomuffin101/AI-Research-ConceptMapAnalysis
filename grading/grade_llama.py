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


class EmptyLlamaVisionResponseError(RuntimeError):
    def __init__(self, message: str, raw_response: Any, attempts: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.attempts = attempts


class MalformedLlamaVisionJsonError(RuntimeError):
    def __init__(self, attempts: dict[str, Any]) -> None:
        super().__init__("Llama 3.2 90B Vision returned malformed or incomplete grading JSON.")
        self.attempts = attempts


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
        + _output_contract()
    )


def _vision_messages(prompt: str, image_base64: str) -> list[dict[str, Any]]:
    """NVIDIA NIM OpenAI-compatible multimodal message: text plus a JPEG data URL."""
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{IMAGE_MIME_TYPE};base64,{image_base64}"}},
    ]}]


def _nvidia_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "messages": messages,
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
    }


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


def request_grade(client: Any, prompt: str, image_base64: str) -> NvidiaChatCompletion:
    return _post_nvidia(client, _nvidia_payload(_vision_messages(prompt, image_base64)))


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
        "choice_text": first.get("text") if isinstance(first, dict) else getattr(first, "text", None),
        "usage": usage,
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
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return json.dumps(value, separators=(",", ":")), value
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


def grade_pdf(
    pdf_path: Path,
    map_file: str,
    debug_prefix: Path,
    reference_materials: list[dict[str, str]] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run exactly one normal single-pass NVIDIA Llama grading request."""
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
    }
    debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
    if progress_callback:
        progress_callback("Grading concept map with Llama 3.2 90B Vision...")
    client = create_client()
    try:
        response, retry_debug = _request_with_retry(
            lambda: request_grade(client, prompt, image_base64), progress_callback
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
    try:
        cleaned_text, parsed = _parse_json_object(raw_text, attempts)
    except RuntimeError as exc:
        debug.update({"complete_response": diagnostics, "http_status": diagnostics["http_status"], "finish_reason": diagnostics["finish_reason"], "raw_path": str(raw_path), "error": str(exc)})
        debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
        raise MalformedLlamaVisionJsonError(attempts) from exc
    parsed_path = Path(f"{debug_prefix}_grading_parsed.json")
    parsed_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    normalizations = _normalize_scores(parsed)
    cleaned_text = json.dumps(parsed, separators=(",", ":"))
    debug.update({
        "request_count": retry_debug["request_count"], "retry_attempted": retry_debug["retry_attempted"],
        "http_status": diagnostics["http_status"], "finish_reason": diagnostics["finish_reason"],
        "usage": diagnostics["usage"], "response_content_length": len(raw_text),
        "complete_response": diagnostics, "raw_path": str(raw_path), "parsed_path": str(parsed_path),
        "score_normalizations": normalizations, "duration_seconds": round(time.monotonic() - started_at, 3),
        "payload_shape": {"messages": [{"role": "user", "content": [{"type": "text"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<image-bytes>"}}]}]},
    })
    debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
    return {"model": MODEL, "provider": PROVIDER, "raw_text": raw_text, "cleaned_text": cleaned_text,
            "response": response, "prompt": prompt, "prompt_path": prompt_path, "image_path": image_path,
            "raw_path": raw_path, "debug": {**debug, "debug_path": str(debug_path)}}
