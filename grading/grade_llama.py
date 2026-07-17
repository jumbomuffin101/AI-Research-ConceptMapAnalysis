"""Direct Groq Llama 4 Scout grader for Spring 2025 concept map evaluation."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from interface.reference_materials import format_reference_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = PROJECT_ROOT / "rubric" / "concept_map_rubric.json"

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
PROVIDER = "Groq"
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_ENV = "GROQ_API_KEY"
MAX_TOKENS = 3000
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

CRITERION_TEXT = {
    "knowledge_acquisition": {
        "basic_science": "Identifies key knowledge from basic sciences learned this unit",
        "health_system_science": "Identifies key knowledge from health system science learned this unit",
        "clinical_science": "Identifies key knowledge from clinical sciences learned this unit",
        "patient_case_information": "Extracts key information from the patient case",
        "determinants_of_health": "Identifies key determinants of health (DoH)",
    },
    "integration": {
        "prioritized_differential_diagnosis": "Includes a prioritized differential diagnosis (DDx) that contains common, must not miss, and other possible diagnoses based on patient’s unique characteristics",
        "illness_scripts": "Connects patient data to reflect illness script(s)",
        "basic_to_foundational_science": "Connects basic science knowledge learned in the unit to other relevant foundational science information",
        "patient_data_to_clinical_information": "Connects patient data to other relevant clinical information",
        "patient_data_to_basic_science": "Connects patient data to relevant basic science knowledge",
    },
    "application": {
        "working_diagnosis_pathophysiology": "Concept map explains the underlying pathophysiology of the working diagnosis",
        "patient_data_pathophysiology": "Connections explain the pathophysiology underlying the key patient data",
    },
    "transfer": {
        "prior_basic_science": "Identifies relevant basic science concepts learned in previous courses",
        "prior_clinical_concepts": "Identifies relevant clinical concepts learned in previous courses",
        "deepens_understanding": "Uses previously learned knowledge to deepen understanding of the pathophysiology of the condition, the “So what?”",
    },
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
    return create_groq_client()


def create_groq_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    api_key = _secret(API_KEY_ENV)
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
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
    rubric_payload = {
        "criteria": CRITERION_TEXT,
        "score_descriptors": _rubric(),
    }
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
    return f"""Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities. Grade visible map only.
{reference_section}
Rubric:{json.dumps(rubric_payload, separators=(",", ":"))}
Rules: scores integers 1-4 only; decisions Yes/No only; no Partial/Borderline/Maybe/0/5/decimals; JSON only.
The final overall decision should reflect the concept map as a whole. A single weak criterion or domain does not automatically require an overall No. Consider whether the map, overall, meets expectations based on the full rubric.
Keep explanation one short sentence.
strengths max 2 short strings; areas_for_improvement max 2 short strings; grading_notes max 1 sentence.
Schema:{json.dumps(schema(map_file), separators=(",", ":"))}
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
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("Llama 4 Scout returned no response choices.")
    text = getattr(choices[0].message, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Llama 4 Scout returned empty content.")
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
    raw_text = response_text(response)
    raw_path = Path(f"{debug_prefix}_raw.txt")
    raw_path.write_text(raw_text, encoding="utf-8")
    debug_payload["raw_path"] = str(raw_path)
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
