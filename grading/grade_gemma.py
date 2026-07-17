"""Direct Gemma grader for Spring 2025 concept map evaluation."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from interface.reference_materials import format_reference_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = PROJECT_ROOT / "rubric" / "concept_map_rubric.json"

MODEL = "google/gemma-4-26b-a4b-it:free"
PROVIDER = "OpenRouter"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
MAX_TOKENS = 1800
TIMEOUT_SECONDS = 90
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


class EmptyGemmaResponseError(RuntimeError):
    """Gemma returned no usable completion content for a request."""

    def __init__(self, message: str, raw_response: Any) -> None:
        super().__init__(message)
        self.raw_response = raw_response


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
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    api_key = _secret(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is not configured.")
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    )


def render_pdf_first_page(pdf_path: Path, output_path: Path) -> str:
    """Render first PDF page to a compressed JPEG and return base64."""
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
    return base64.b64encode(image_bytes).decode("utf-8")


def _rubric() -> dict[str, Any]:
    rubric_data = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    return {
        group: rubric_data[group]
        for group in CATEGORY_FIELDS
        if isinstance(rubric_data.get(group), dict)
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
    reference_context = format_reference_context(reference_materials or [])
    reference_section = (
        f"""REFERENCE MATERIAL
The following files define the patient case and/or course content the student was expected to use.

{reference_context}

STUDENT CONCEPT MAP
The concept map image is the only source of evidence for what the student actually included.

Instructions for reference use:
- Use the reference material only as the comparison standard.
- Score only content visibly present in the student concept map.
- Do not treat reference-material content as if it appears in the map.
- For learned-this-unit criteria, compare visible map content against uploaded unit/session material.
- For patient-case completeness, compare visible map content against the uploaded patient case.
- For DDx, compare the map against clinically relevant alternatives supported by the reference materials.
- Do not require every reference detail unless required by the rubric.
- Do not lower scores solely because expected reference information is absent from the uploaded files.

"""
        if reference_context
        else ""
    )
    return f"""Use the Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities exactly.
Grade only the visible concept map image.

{reference_section}Rubric JSON:
{json.dumps(_rubric(), separators=(",", ":"))}

Rules:
- Grade strictly according to the exact Spring 2025 rubric descriptors.
- Judge relevance, detail, synthesis, visible connections, and comprehensiveness.
- Medically correct words or isolated concepts are not sufficient for a high score.
- Do not reward content that could be inferred but is not visibly represented in the concept map.
- Scores must be integers 1, 2, 3, or 4 only.
- Overall decisions must be exactly Yes or No only.
- No Partial, Borderline, Maybe, 0, 5, or decimals.
- Use brief explanations only.

Score calibration:
- Score 1: the criterion is absent, irrelevant, unclear, or does not meaningfully address the criterion; required connections are absent or unclear.
- Score 2: some relevant information is present, but it is general, incomplete, simplistic, weakly connected, or only an isolated label/list without synthesis.
- Score 3: relevant information is clearly present, mostly synthesized, and required connections are meaningfully demonstrated; it satisfies most but not all of the score-4 descriptor.
- Score 4: reserve for clear fulfillment of the full descriptor with detailed, synthesized, comprehensive content. Required breadth across sessions, diagnoses, components, or map areas must be visibly demonstrated.
- When uncertain between two scores, choose the lower score unless the higher descriptor is clearly satisfied.

Domain calibration:
- Knowledge Acquisition: do not give high scores for terminology alone. Distinguish isolated facts from knowledge meaningfully connected across the map. Do not infer Health System Science or Determinants of Health from generic management or patient information. Patient Case Information must be relevant and synthesized; copied facts alone are not automatically comprehensive.
- Integration: requires visible relationships. Two nearby concepts do not demonstrate integration unless their relationship is visibly represented. A score-4 DDx must be explicitly prioritized; an unranked list is not prioritized. Illness scripts and science/clinical integration require visible logical flow, not proximity.
- Application: requires visible causal or mechanistic pathophysiology flows. Listing a diagnosis, symptoms, labs, or treatments is not pathophysiological application. Score 3 or 4 only when explanatory flows are clearly demonstrated.
- Transfer: award credit only when previously learned concepts are visibly used to deepen understanding of the current condition. Do not assume a concept is prior knowledge merely because it is basic or clinical science. Generic medical knowledge without deeper application scores 1 or 2.

Reference-material rules:
- If reference materials are uploaded, use them only to understand expected knowledge and available patient-case information. Never give credit for content that appears only in those files or assume the student included it. Missing expected content may reduce a score under the rubric, but do not require irrelevant reference details outside the rubric.
- If no reference materials are uploaded, grade normally from the visible concept map and rubric; do not penalize because references are unavailable.

Domain decisions:
- Knowledge Acquisition is Yes only when the map meaningfully includes key knowledge from the case and unit.
- Integration is Yes only when key knowledge is connected accurately and comprehensively.
- Application is Yes only when key clinical data is explained using relevant basic science.
- Transfer is Yes only when previously learned content visibly deepens understanding.
- A major missing rubric component requires No even when other criteria are strong. Do not infer a domain decision from average score alone.

Final decision and consistency review:
- This map meets expectations is Yes only when performance is adequate across the rubric as a whole; a few high scores cannot compensate for major missing domains.
- The final overall decision should reflect the concept map as a whole. A single weak criterion or domain does not automatically require an overall No. Consider whether the map, overall, meets expectations based on the full rubric.
- Before returning JSON, re-check every criterion against its descriptor, verify every 4 satisfies the full score-4 descriptor, verify isolated content was not over-scored, verify Integration/Application reflect visible connections, and verify domain and final decisions are defensible across all four domains.
- Return JSON only using this exact schema:
{json.dumps(schema(map_file), separators=(",", ":"))}
"""


def request_grade(client: Any, prompt: str, image_base64: str) -> Any:
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        timeout=TIMEOUT_SECONDS,
        response_format={"type": "json_object"},
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


def response_text(response: Any) -> str:
    if response is None:
        raise EmptyGemmaResponseError("Gemma returned no response.", response)
    choices = getattr(response, "choices", None)
    if choices is None:
        raise EmptyGemmaResponseError("Gemma response has no choices.", response)
    if not choices:
        raise EmptyGemmaResponseError("Gemma returned no response choices.", response)
    message = getattr(choices[0], "message", None)
    if message is None:
        raise EmptyGemmaResponseError("Gemma response choice has no message.", response)
    text = getattr(message, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise EmptyGemmaResponseError("Gemma returned empty content.", response)
    return text


def _response_debug_value(response: Any) -> Any:
    """Create a JSON-serializable record without exposing credentials."""
    if response is None:
        return None
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except Exception:
            pass
    return repr(response)


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
    image_base64 = render_pdf_first_page(pdf_path, image_path)
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

    client = create_client()
    response = request_grade(client, prompt, image_base64)
    attempts_path = Path(f"{debug_prefix}_response_attempts.json")
    attempts: dict[str, Any] = {"first_attempt": _response_debug_value(response)}
    try:
        raw_text = response_text(response)
    except EmptyGemmaResponseError as first_error:
        attempts_path.write_text(json.dumps(attempts, indent=2), encoding="utf-8")
        time.sleep(5)
        retry_response = request_grade(client, prompt, image_base64)
        attempts["retry_attempt"] = _response_debug_value(retry_response)
        attempts_path.write_text(json.dumps(attempts, indent=2), encoding="utf-8")
        try:
            raw_text = response_text(retry_response)
        except EmptyGemmaResponseError as retry_error:
            raise EmptyGemmaResponseError(str(retry_error), retry_response) from first_error
        response = retry_response

    raw_path = Path(f"{debug_prefix}_raw.txt")
    raw_path.write_text(raw_text, encoding="utf-8")

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
            "provider": PROVIDER,
            "base_url": BASE_URL,
            "model": MODEL,
            "image_path": str(image_path),
            "image_bytes": image_path.stat().st_size,
            "reference_materials_used": bool(reference_files),
            "reference_files": reference_files,
            "empty_response_retry_attempted": "retry_attempt" in attempts,
            "response_attempts_path": str(attempts_path) if attempts_path.exists() else None,
            "first_attempt": attempts["first_attempt"],
            "retry_attempt": attempts.get("retry_attempt"),
            "max_tokens": MAX_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
    }
