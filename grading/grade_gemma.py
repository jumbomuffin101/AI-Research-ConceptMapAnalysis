"""Direct Gemma grader for Spring 2025 concept map evaluation."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = PROJECT_ROOT / "rubric" / "concept_map_rubric.json"

MODEL = "google/gemma-4-26b-a4b-it:free"
PROVIDER = "OpenRouter"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
MAX_TOKENS = 1800
TIMEOUT_SECONDS = 90
NO_REFERENCE_WARNING = (
    "Scores involving unit coverage, patient-case completeness, or prior-course "
    "knowledge are provisional because no reference materials were supplied."
)
REFERENCE_FIELDS = (
    ("patient_case", "Patient case"),
    ("unit_content", "Unit learning objectives or session content"),
    ("expected_differential_diagnoses", "Expected/key differential diagnoses"),
    ("prior_concepts", "Relevant previously learned concepts"),
    ("instructor_notes", "Instructor notes or expected content"),
)

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
            field: {"score": 1, "explanation": "", "evidence_from_map": []}
            for field in fields
        }
        result[group]["overall_decision"] = "No"
        result[group]["if_no_explanation"] = ""
    result["overall_meets_expectations"] = "No"
    result["strengths"] = ["", ""]
    result["areas_for_improvement"] = ["", ""]
    result["grading_notes"] = ""
    return result


def _format_reference_material(
    reference_material: dict[str, str] | None,
) -> tuple[str, bool]:
    material = reference_material or {}
    sections: list[str] = []
    for key, label in REFERENCE_FIELDS:
        value = str(material.get(key, "")).strip()
        if value:
            sections.append(f"{label}:\n{value}")
    if not sections:
        return f"No reference material supplied.\nWARNING: {NO_REFERENCE_WARNING}", False
    return "\n\n".join(sections), True


def build_prompt(map_file: str, reference_material: dict[str, str] | None = None) -> str:
    reference_text, has_reference = _format_reference_material(reference_material)
    no_reference_rule = (
        ""
        if has_reference
        else f'\n- Include this exact warning in grading_notes: "{NO_REFERENCE_WARNING}"'
    )
    return f"""Use the Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities exactly.

REFERENCE MATERIAL:
{reference_text}

STUDENT CONCEPT MAP:
The uploaded image is the student's concept map.

Rubric JSON:
{json.dumps(_rubric(), separators=(",", ":"))}

Rules:
- Grade only what appears in the STUDENT CONCEPT MAP image.
- Use REFERENCE MATERIAL only to determine what content was expected.
- Do not treat REFERENCE MATERIAL text as evidence that appears in the map.
- evidence_from_map must contain only content visible in the STUDENT CONCEPT MAP.
- Missing expected content should reduce the relevant score according to the rubric.
- Do not require content that is not present in the supplied REFERENCE MATERIAL.
- Scores must be integers 1, 2, 3, or 4 only.
- Overall decisions must be exactly Yes or No only.
- No Partial, Borderline, Maybe, 0, 5, or decimals.
- Use brief explanations only.
- Include evidence_from_map when visible.
- If evidence is missing, write "No clear evidence found in the concept map."
- Do not hallucinate evidence.
- Grade only demonstrated content and visible relationships.
- Do not reward the presence of isolated medical terms.
- Do not infer missing knowledge from general medical plausibility.
- A correct term without synthesis or connection does not satisfy a score-3 or score-4 descriptor.
- Score 4 only when the full score-4 descriptor is clearly satisfied.
- Score 3 only when evidence is relevant and mostly synthesized.
- Score 2 when evidence is partly relevant, too general, incomplete, or weakly connected.
- Score 1 when evidence is absent, irrelevant, or unclear.
- Knowledge Acquisition overall_decision is Yes only when the map demonstrates sufficient key knowledge from the case and unit content across the domain. A few strong criteria must not compensate for major missing areas.
- Integration overall_decision is Yes only when key knowledge is connected accurately and comprehensively. Isolated concepts or weak connections are insufficient.
- Application overall_decision is Yes only when the map clearly explains key clinical data using relevant basic science and visible pathophysiology flows.
- Transfer overall_decision is Yes only when previously learned content visibly deepens understanding of the current condition.
- overall_meets_expectations must be "Yes" only when all four domain overall_decision values are "Yes".
- If any domain overall_decision is "No", overall_meets_expectations must be "No".
- A weak map must not pass because it contains some correct concepts.
- Before returning JSON, perform a consistency check: each domain decision must agree with its criterion scores and evidence, the final overall decision must agree with all four domain decisions, and any No domain requires final overall No.
- Keep REFERENCE MATERIAL and STUDENT CONCEPT MAP evidence separate.{no_reference_rule}
- Return JSON only using this exact schema:
{json.dumps(schema(map_file), separators=(",", ":"))}
"""


def request_grade(client: Any, prompt: str, image_base64: str) -> Any:
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


def response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Gemma returned no response choices.")
    text = getattr(choices[0].message, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Gemma returned empty content.")
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
    reference_material: dict[str, str] | None = None,
) -> dict[str, Any]:
    image_path = Path(f"{debug_prefix}_request.jpg")
    image_base64 = render_pdf_first_page(pdf_path, image_path)
    prompt = build_prompt(map_file, reference_material)
    prompt_path = Path(f"{debug_prefix}_prompt.txt")
    prompt_path.write_text(prompt, encoding="utf-8")
    _, has_reference_material = _format_reference_material(reference_material)

    client = create_client()
    response = request_grade(client, prompt, image_base64)
    raw_text = response_text(response)
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
            "reference_material_supplied": has_reference_material,
            "max_tokens": MAX_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
    }
