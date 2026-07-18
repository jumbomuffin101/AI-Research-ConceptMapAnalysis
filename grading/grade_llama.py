"""Direct OpenRouter Nemotron grader for Spring 2025 concept map evaluation."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from grading.spring_2025_prompt import build_grading_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
PROVIDER = "OpenRouter"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
MAX_TOKENS = 1800
TIMEOUT_SECONDS = 180
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


class EmptyNemotronResponseError(RuntimeError):
    """Nemotron returned no usable completion content."""

    def __init__(self, message: str, raw_response: Any, attempts: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.attempts = attempts


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
    return create_openrouter_client()


def create_openrouter_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    api_key = _secret(API_KEY_ENV)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )


def render_pdf_first_page(pdf_path: Path, output_path: Path) -> dict[str, Any]:
    """Render first PDF page to a small compressed JPEG."""
    import fitz

    with fitz.open(pdf_path) as document:
        if document.page_count < 1:
            raise RuntimeError("The uploaded PDF has no pages.")
        page = document[0]
        max_width_px = 1800
        scale = max_width_px / max(page.rect.width, 1)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image_bytes = pixmap.tobytes("jpeg", jpg_quality=85)
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
        "jpeg_quality": 85,
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
    return build_grading_prompt(map_file, schema(map_file), reference_materials)


def request_grade(client: Any, prompt: str, image_base64: str) -> Any:
    """Request normal text content for local JSON parsing and validation."""
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        timeout=TIMEOUT_SECONDS,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
    )


def _response_debug_value(response: Any) -> Any:
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            pass
    return repr(response)


def response_text(response: Any, attempts: dict[str, Any]) -> str:
    if response is None:
        raise EmptyNemotronResponseError("Nemotron returned no response.", response, attempts)
    choices = getattr(response, "choices", None)
    if not choices:
        raise EmptyNemotronResponseError("Nemotron returned no response choices.", response, attempts)
    message = getattr(choices[0], "message", None)
    text = getattr(message, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise EmptyNemotronResponseError("Nemotron returned empty content.", response, attempts)
    return text


def clean_json_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0).strip() if match else text


def grade_pdf(
    pdf_path: Path,
    map_file: str,
    debug_prefix: Path,
    reference_materials: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    image_path = Path(f"{debug_prefix}_request.jpg")
    image_info = render_pdf_first_page(pdf_path, image_path)
    image_base64 = str(image_info["base64"])
    prompt = build_prompt(map_file, reference_materials)
    prompt_path = Path(f"{debug_prefix}_prompt.txt")
    reference_files = [item["filename"] for item in reference_materials or []]
    if reference_files:
        prompt_path.write_text(
            "Reference text omitted from debug output. Files used: "
            + ", ".join(reference_files)
            + "\n\n"
            + build_prompt(map_file),
            encoding="utf-8",
        )
    else:
        prompt_path.write_text(prompt, encoding="utf-8")
    debug_path = Path(f"{debug_prefix}_debug.json")
    debug_payload = {
        "provider": PROVIDER,
        "base_url": BASE_URL,
        "model": MODEL,
        "image_path": str(image_info["path"]),
        "image_width": image_info["width"],
        "image_height": image_info["height"],
        "image_bytes": image_info["bytes"],
        "render_matrix": image_info["render_matrix"],
        "max_width_px": image_info["max_width_px"],
        "jpeg_quality": image_info["jpeg_quality"],
        "reference_materials_used": bool(reference_files),
        "reference_files": reference_files,
        "prompt_characters": len(prompt),
        "max_tokens": MAX_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    client = create_client()
    response = request_grade(client, prompt, image_base64)
    attempts = {"first_attempt": _response_debug_value(response)}
    try:
        raw_text = response_text(response, attempts)
    except EmptyNemotronResponseError as first_error:
        time.sleep(5)
        retry_response = request_grade(client, prompt, image_base64)
        attempts["retry_attempt"] = _response_debug_value(retry_response)
        try:
            raw_text = response_text(retry_response, attempts)
        except EmptyNemotronResponseError as retry_error:
            raise EmptyNemotronResponseError(str(retry_error), retry_response, attempts) from first_error
        response = retry_response
    raw_path = Path(f"{debug_prefix}_raw.txt")
    raw_path.write_text(raw_text, encoding="utf-8")
    debug_payload["raw_path"] = str(raw_path)
    debug_payload["first_attempt"] = attempts["first_attempt"]
    debug_payload["retry_attempt"] = attempts.get("retry_attempt")
    debug_payload["empty_response_retry_attempted"] = "retry_attempt" in attempts
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    return {
        "model": MODEL,
        "provider": PROVIDER,
        "raw_text": raw_text,
        "cleaned_text": clean_json_output(raw_text),
        "response": response,
        "prompt": prompt,
        "prompt_path": prompt_path,
        "image_path": image_path,
        "raw_path": raw_path,
        "debug": {
            **debug_payload,
            "debug_path": str(debug_path),
            "raw_path": str(raw_path),
        },
    }
