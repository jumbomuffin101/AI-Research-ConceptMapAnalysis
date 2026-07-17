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
    reference_material: dict[str, Any] | None,
) -> tuple[str, bool]:
    material = reference_material or {}
    if not material:
        return "", False
    return json.dumps(material, ensure_ascii=False, separators=(",", ":")), True


def build_prompt(map_file: str, reference_material: dict[str, Any] | None = None) -> str:
    reference_text, has_reference = _format_reference_material(reference_material)
    reference_section = (
        f"""REFERENCE MATERIAL:
{reference_text}

Use REFERENCE MATERIAL only to determine expected content. Do not treat it as map evidence. Do not require content absent from supplied references.
For learned-this-unit criteria, compare visible map content with the supplied unit reference. For patient-case criteria, compare visible map content with the supplied case. For DDx and transfer, use the reference only to identify relevant expected alternatives or prior concepts.

"""
        if has_reference
        else ""
    )
    return f"""Use the Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities exactly.

STUDENT CONCEPT MAP:
The uploaded image is the student's concept map.

{reference_section}Rubric JSON:
{json.dumps(_rubric(), separators=(",", ":"))}

Rules:
- Grade only demonstrated content and visible relationships in the STUDENT CONCEPT MAP image.
- Grade only the visible concept map against the rubric descriptors when no reference material is supplied.
- Score only content actually visible in the student concept map; reference material is never map evidence.
- evidence_from_map must contain only specific content visible in the student concept map.
- Every numeric criterion must include evidence_from_map as a JSON list of 1-3 short strings, or [] when no supporting evidence is visible.
- Do not infer content that is not visible.
- Do not infer missing knowledge from general medical plausibility.
- Do not reward isolated medical terms as synthesized knowledge.
- Score 1: criterion is absent, irrelevant, or visibly incorrect.
- Score 2: some relevant content exists, but it is general, incomplete, simplistic, or weakly connected.
- Score 3: content is relevant and mostly synthesized, with meaningful visible connections when required.
- Score 4: the complete score-4 descriptor is visibly demonstrated with detailed, comprehensive, synthesized content.
- Do not assign 1 merely because reference context was not supplied.
- Visible relevant concepts should receive at least 2 when they partially address the criterion.
- Use the exact Spring 2025 rubric descriptors as the final authority.
- Scores must be integers 1, 2, 3, or 4 only.
- Overall decisions must be exactly Yes or No only.
- No Partial, Borderline, Maybe, 0, 5, or decimals.
- Explanations: one brief sentence.
- evidence_from_map: 1-3 short items per criterion, or [] if no evidence is visible.
- strengths: maximum 2 short items.
- areas_for_improvement: maximum 2-3 short items.
- grading_notes: maximum one sentence.
- If any domain overall_decision is "No", overall_meets_expectations must be "No".
- Return raw valid JSON only. No Markdown, no prose outside JSON.
- Use this exact schema:
{json.dumps(schema(map_file), separators=(",", ":"))}
"""


def build_repair_prompt(map_file: str, malformed_output: str) -> str:
    return f"""Your previous answer was malformed JSON. Do not regrade or change the evaluation.
Return the same evaluation as raw valid JSON only. No Markdown. No prose.

Required schema:
{json.dumps(schema(map_file), separators=(",", ":"))}

Malformed output:
{malformed_output}
"""


def _json_is_malformed(text: str) -> bool:
    try:
        json.loads(clean_json_output(text))
    except json.JSONDecodeError:
        return True
    return False


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


def request_repair(client: Any, prompt: str) -> Any:
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        timeout=TIMEOUT_SECONDS,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
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
    reference_material: dict[str, Any] | None = None,
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
    first_raw_path = None
    repair_prompt_path = None
    repair_raw_path = None
    repaired = False

    if _json_is_malformed(raw_text):
        first_raw_path = Path(f"{debug_prefix}_raw_malformed_first.txt")
        first_raw_path.write_text(raw_text, encoding="utf-8")
        repair_prompt = build_repair_prompt(map_file, raw_text)
        repair_prompt_path = Path(f"{debug_prefix}_repair_prompt.txt")
        repair_prompt_path.write_text(repair_prompt, encoding="utf-8")
        repair_response = request_repair(client, repair_prompt)
        raw_text = response_text(repair_response)
        repair_raw_path = Path(f"{debug_prefix}_raw_repair.txt")
        repair_raw_path.write_text(raw_text, encoding="utf-8")
        raw_path = repair_raw_path
        response = repair_response
        repaired = True

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
            "json_repair_attempted": repaired,
            "first_malformed_raw_path": str(first_raw_path) if first_raw_path else None,
            "repair_prompt_path": str(repair_prompt_path) if repair_prompt_path else None,
            "repair_raw_path": str(repair_raw_path) if repair_raw_path else None,
            "max_tokens": MAX_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
    }
