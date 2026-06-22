"""Upload-driven grading pipeline used by the Streamlit demo.

The command-line graders remain independent. This module mirrors their proven
PDF rendering and OpenRouter request flow while accepting any uploaded PDF.
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

import fitz
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = PROJECT_ROOT / "rubric" / "concept_map_rubric.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web_demo"

MODEL_CONFIGS = {
    "Qwen": {
        "model_id": "qwen/qwen2.5-vl-72b-instruct",
        "max_tokens": 8000,
    },
    "Nemotron": {
        "model_id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "max_tokens": 8000,
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


class GradingError(RuntimeError):
    """A user-facing failure while grading a concept map."""


class InvalidPDFError(GradingError):
    """The uploaded file cannot be read as a PDF."""


class ModelResponseError(GradingError):
    """A model returned no usable grading response."""


class MalformedResultError(GradingError):
    """A model response is not valid grading JSON."""


@dataclass(frozen=True)
class EvaluationResult:
    """One model's parsed result and persisted output location."""

    model_name: str
    model_id: str
    data: dict[str, Any]
    output_path: Path


def selected_model_names(selection: str) -> list[str]:
    """Translate the UI selection into model registry keys."""
    if selection == "Both":
        return ["Qwen", "Nemotron"]
    if selection not in MODEL_CONFIGS:
        raise GradingError(f"Unknown model selection: {selection}")
    return [selection]


def render_pdf_pages(pdf_path: Path, scale: float = 2.0) -> list[str]:
    """Render all PDF pages as base64 PNG strings for vision input."""
    try:
        with fitz.open(pdf_path) as document:
            if document.page_count < 1:
                raise InvalidPDFError("The uploaded PDF has no pages.")

            images = []
            matrix = fitz.Matrix(scale, scale)
            for page in document:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                images.append(base64.b64encode(pixmap.tobytes("png")).decode("ascii"))
            return images
    except InvalidPDFError:
        raise
    except (fitz.FileDataError, fitz.EmptyFileError, RuntimeError, ValueError) as exc:
        raise InvalidPDFError("The uploaded file is not a valid, readable PDF.") from exc


def load_rubric() -> dict[str, Any]:
    """Load and validate the local rubric JSON."""
    try:
        data = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GradingError(f"Rubric not found at {RUBRIC_PATH}.") from exc
    except json.JSONDecodeError as exc:
        raise GradingError("The rubric file contains invalid JSON.") from exc
    if not isinstance(data, dict):
        raise GradingError("The rubric must contain a JSON object.")
    return data


def build_prompt(rubric: dict[str, Any], map_file: str, model_id: str) -> str:
    """Build the grounded grading prompt shared by every registered model."""
    schema: dict[str, Any] = {"map_file": map_file, "model": model_id}
    for group, fields in CATEGORY_FIELDS.items():
        schema[group] = {
            **{
                field: {"score": 0, "reasoning": "", "evidence_from_map": []}
                for field in fields
            },
            "overall": {"meets_expectations": "", "reasoning": ""},
        }
    schema.update(
        {
            "overall_map_meets_expectations": "",
            "strengths": [{"description": "", "evidence_from_map": []}],
            "areas_for_improvement": [
                {"description": "", "missing_or_weak_evidence": []}
            ],
            "grading_notes": "",
        }
    )

    return f"""You are evaluating a student medical concept map using the rubric below.

Use only evidence directly visible in the concept map. For every scored category,
include short, directly traceable visible phrases in evidence_from_map. If no direct
evidence is visible, use [\"No direct supporting evidence visible.\"] and reduce the
score. A score of 3 or 4 requires at least two pieces of direct supporting evidence.

Rubric:
{json.dumps(rubric, indent=2)}

Return ONLY raw valid JSON matching this structure exactly:
{json.dumps(schema, indent=2)}

Rules:
- Every numeric score must be an integer from 1 to 4.
- Use the rubric definitions as the source of truth.
- Do not add markdown fences or text outside the JSON object.
- Evaluate every PDF page supplied in the image inputs.
"""


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Extract, parse, and minimally validate grading JSON from a model response."""
    if not raw_text or not raw_text.strip():
        raise ModelResponseError("The model returned an empty response.")

    text = re.sub(r"^\s*```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("{")
    if start < 0:
        raise MalformedResultError("The model response did not contain a JSON object.")

    try:
        result, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise MalformedResultError(
            f"The model returned malformed JSON ({exc.msg}, line {exc.lineno})."
        ) from exc

    if not isinstance(result, dict):
        raise MalformedResultError("The model result must be a JSON object.")

    missing = [
        key
        for key in (*CATEGORY_FIELDS.keys(), "overall_map_meets_expectations")
        if key not in result
    ]
    if missing:
        raise MalformedResultError(
            "The model result is missing required fields: " + ", ".join(missing)
        )

    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            raise MalformedResultError(f"'{group}' must be a JSON object.")
        for field in fields:
            item = section.get(field)
            score = item.get("score") if isinstance(item, dict) else None
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 4:
                raise MalformedResultError(
                    f"'{group}.{field}.score' must be an integer from 1 to 4."
                )
    return result


def _create_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise GradingError(
            "OPENROUTER_API_KEY is missing. Add it to the environment or project .env file."
        )
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=300)


def _request_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    page_images: list[str],
) -> tuple[str, str]:
    config = MODEL_CONFIGS[model_name]
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
        }
        for image in page_images
    )

    try:
        response = client.chat.completions.create(
            model=config["model_id"],
            max_tokens=config["max_tokens"],
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        raise ModelResponseError(f"{model_name} request failed: {exc}") from exc

    if not response.choices:
        raise ModelResponseError(f"{model_name} returned no response choices.")
    text = response.choices[0].message.content
    if not isinstance(text, str) or not text.strip():
        raise ModelResponseError(f"{model_name} returned no usable content.")
    return config["model_id"], text


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned[:60] or "concept_map"


def run_evaluation(
    pdf_path: Path,
    model_names: Iterable[str],
    original_filename: str,
) -> list[EvaluationResult]:
    """Grade an uploaded PDF with each selected model and persist valid JSON."""
    names = list(model_names)
    if not names:
        raise GradingError("Select at least one model.")
    unknown = [name for name in names if name not in MODEL_CONFIGS]
    if unknown:
        raise GradingError("Unknown model(s): " + ", ".join(unknown))

    # Render before creating the API client so invalid uploads fail without a request.
    page_images = render_pdf_pages(pdf_path)
    rubric = load_rubric()
    client = _create_client()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex[:8]
    file_stem = _safe_stem(original_filename)
    results = []

    for model_name in names:
        model_id = MODEL_CONFIGS[model_name]["model_id"]
        prompt = build_prompt(rubric, Path(original_filename).name, model_id)
        returned_model_id, raw_text = _request_model(
            client, model_name, prompt, page_images
        )
        data = parse_model_json(raw_text)
        output_path = OUTPUT_DIR / (
            f"{timestamp}_{run_id}_{file_stem}_{model_name.lower()}.json"
        )
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        results.append(
            EvaluationResult(model_name, returned_model_id, data, output_path)
        )

    return results

