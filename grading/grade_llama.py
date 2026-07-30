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
FULL_RETRY_MAX_TOKENS = 2600
TEMPERATURE = 0.2
TOP_P = 0.9
TIMEOUT_SECONDS = 120
CONNECT_TIMEOUT_SECONDS = 30
FULL_RETRY_READ_TIMEOUT_SECONDS = 300
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

RETRY_CRITERION_LABELS = {
    "knowledge_acquisition": {
        "basic_science": "Identifies key knowledge from basic sciences learned this unit",
        "health_system_science": "Identifies key knowledge from health system science learned this unit",
        "clinical_science": "Identifies key knowledge from clinical sciences learned this unit",
        "patient_case_information": "Extracts key information from the patient case",
        "determinants_of_health": "Identifies key determinants of health (DoH)",
        "overall": "Does the student's map include key knowledge from the case and content learned during this unit?",
    },
    "integration": {
        "prioritized_differential_diagnosis": "Includes a prioritized differential diagnosis (DDx)",
        "illness_scripts": "Connects patient data to reflect illness script(s)",
        "basic_to_foundational_science": "Connects basic science knowledge to foundational science information",
        "patient_data_to_clinical_information": "Connects patient data to relevant clinical information",
        "patient_data_to_basic_science": "Connects patient data to relevant basic science knowledge",
        "overall": "Did the learner connect key knowledge accurately and comprehensively?",
    },
    "application": {
        "working_diagnosis_pathophysiology": "Explains pathophysiology of the working diagnosis",
        "patient_data_pathophysiology": "Explains pathophysiology underlying key patient data",
        "overall": "Did the learner explain key clinical data with relevant basic science?",
    },
    "transfer": {
        "prior_basic_science": "Identifies relevant previous-course basic science concepts",
        "prior_clinical_concepts": "Identifies relevant previous-course clinical concepts",
        "deepens_understanding": "Uses previous knowledge to deepen understanding of pathophysiology",
        "overall": "Did the learner use previously learned content to deepen understanding?",
    },
}


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
        "strengths": ["<concise evidence-based strength>"],
        "areas_for_improvement": ["<concise evidence-based improvement area>"],
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
        "\nSCORING CALIBRATION FOR STRONG MAPS\n"
        "Score only visible map evidence. A score of 4 does not require perfection, exhaustive detail, or "
        "every possible concept. Award 4 when the map clearly and accurately demonstrates the full intent "
        "of the rubric criterion through sufficiently detailed and integrated visible evidence. Minor "
        "omissions, wording differences, or opportunities for added detail do not automatically reduce a "
        "fully demonstrated criterion from 4 to 3.\n"
        "Use 3 only when the criterion is substantially demonstrated but a meaningful visible limitation "
        "in completeness, accuracy, specificity, prioritization, integration, patient-data connection, or "
        "connection between concepts prevents full demonstration. Before choosing 3 instead of 4, identify "
        "that specific limitation. Do not choose 3 merely because more detail could theoretically be added.\n"
        "Keep the lower scores strict: use 2 for partial, superficial, inconsistent, or weakly connected "
        "evidence; use 1 for absent, largely incorrect, or unsupported evidence. Many words alone do not "
        "show integration or application. Do not infer missing content from reference material. Reference "
        "materials identify major expected concepts and relationships, not an exhaustive standard; never "
        "treat reference text as map evidence.\n"
        "Each explanation must be concise and evidence-specific. For 4, name the visible concepts, "
        "relationships, patient findings, diagnostic structure, or pathophysiologic links that fully "
        "demonstrate the criterion. For 3, name both what the map demonstrates well and the concrete, "
        "meaningful visible limitation preventing 4. Do not use generic phrases such as 'could include more "
        "detail', 'good but not perfect', or 'not fully comprehensive' without a specific omission.\n"
        "\nCALIBRATION EXAMPLES (illustrative only; grade the submitted map independently)\n"
        "Example A — score 4: a map accurately connects patient symptoms, exam findings, mechanism, "
        "diagnosis, treatment, and relevant risk factors in a clear integrated pathway. Some secondary "
        "details are absent, but the full criterion intent is demonstrated.\n"
        "Example B — score 3: a map contains correct patient findings and diagnosis, but lists them "
        "separately without clearly connecting them to the underlying mechanism.\n"
        "Before returning JSON, silently review every score of 3: retain it only when its explanation "
        "identifies a meaningful visible deficiency; otherwise use 4 if the criterion's full intent is "
        "clearly demonstrated. For every 4, verify the map visibly demonstrates the full criterion intent "
        "without requiring perfection. Do not output this review.\n\n"
        "OVERALL DECISION CALIBRATION\n"
        "The final Yes/No decision is a holistic judgment of whether the concept map meets the educational "
        "expectations of the rubric. It is not a requirement for every criterion to receive 4, every domain "
        "to be flawless, a mathematical threshold or average, or an automatic failure because one or more "
        "criteria receive 3 or have minor omissions. Several 3s alone do not decide the result, but the map "
        "must still demonstrate the required relationships and reasoning in central domains.\n"
        "Return Yes when the visible map as a whole demonstrates central clinical and scientific concepts, "
        "meaningfully connects patient data to diagnoses, mechanisms, or management, shows substantial "
        "integration rather than isolated fact listing, applies knowledge to the working diagnosis or key "
        "patient findings, and contains enough accurate, connected evidence to satisfy the rubric's main intent. "
        "Minor or moderate limitations do not require No only when they do not undermine central educational performance.\n"
        "Return No only for a substantial map-level deficiency that prevents the map meeting educational "
        "expectations: for example, absent major domains, missing central concepts, mostly disconnected patient "
        "data, absent or seriously incorrect pathophysiology, a list of terms without meaningful relationships, "
        "multiple weak core criteria, or inaccuracies that substantially undermine reasoning. Before No, identify "
        "the major unmet expectation, visible evidence of that failure, and why it outweighs the rest of the map. "
        "Do not use generic incompleteness, several 3s, or opportunities for improvement as a failure basis.\n"
        "Apply the same principle to each domain: answer Yes when its central purpose is clearly demonstrated "
        "with mostly accurate, meaningfully connected evidence despite minor or moderate limitations; answer No "
        "only when the domain's central purpose is not sufficiently demonstrated. Do not require every domain "
        "criterion to receive 4.\n"
        "A score of 4 represents full criterion intent at the expected student level, not expert depth, every "
        "possible fact, exhaustive reference coverage, perfect wording, every relationship label, or no remaining "
        "improvement opportunity. A successful map may still have areas_for_improvement; that feedback does not "
        "imply overall No.\n"
        "\nOVERALL CALIBRATION EXAMPLES (illustrative only; grade the submitted map independently)\n"
        "Example A — strong map that passes: a map accurately connects patient findings, differential diagnoses, "
        "working diagnosis, pathophysiology, treatment, and relevant foundational science. Some secondary "
        "relationships are not fully labeled and several criteria receive 3. Correct overall decision: Yes, "
        "because major educational objectives are demonstrated with only moderate limitations.\n"
        "Example B — weak map that fails: a map lists symptoms, diagnoses, and treatments without connections, "
        "has little or no pathophysiology, and omits major patient-specific reasoning. Correct overall decision: "
        "No, because central integration and application expectations are not demonstrated.\n"
        "Before returning JSON, silently review every domain No and final No: identify a specific substantial "
        "visible deficiency affecting a central educational expectation, and confirm the decision is not based "
        "only on 3s, minor omissions, or improvement opportunities. Also verify that Integration and Application "
        "are meaningfully demonstrated rather than merely listed. Do not output this review.\n\n"
        "EVIDENCE SUFFICIENCY\n"
        "A concept or term being present does not by itself demonstrate a rubric criterion. Credit the requested "
        "relationship, explanation, prioritization, or application—not keyword presence. Listing diagnoses is not "
        "a prioritized differential; listing symptoms and a diagnosis is not clinical reasoning; listing anatomy or "
        "physiology is not linking patient data to basic science; naming a mechanism is not explaining "
        "pathophysiology; listing social factors is not explaining their effect; and mentioning prior knowledge is "
        "not transfer. Evidence must be visibly demonstrated in the concept map. Do not infer relationships because "
        "concepts are near one another; do not assume prioritization because one diagnosis is present; do not assume "
        "illness scripts because symptoms and a diagnosis appear together; do not assume integration merely because "
        "arrows exist; and do not assume pathophysiologic reasoning merely because a mechanism is named. A high score "
        "requires an explicit or clearly demonstrated required relationship. When uncertain whether a relationship is "
        "demonstrated, use the lower score.\n"
        "For Integration, Application, and Transfer, do not award 3 or 4 unless meaningful relationships are visibly "
        "shown by arrows, connecting lines, linking phrases, causal sequences, hierarchy, explicit grouping, or "
        "patient-specific explanatory statements. Concepts placed near each other without a clear relationship do "
        "not automatically earn integration credit. Do not inflate a score because the map is visually dense or has "
        "many medically relevant words.\n"
        "Use these operational definitions: 4 requires the full criterion intent with accurate, specific, meaningfully "
        "connected evidence; 3 requires most of the criterion with one meaningful limitation; 2 requires partial "
        "evidence with limited, superficial, generic, inconsistent, or incomplete connections; 1 is absent, seriously "
        "incorrect, unsupported, or only isolated terms with no meaningful demonstration. For every 3 or 4, the "
        "explanation must name the specific content and relationship or application that justifies that level.\n"
        "Criterion guardrails: prioritized DDx requires multiple plausible diagnoses ranked or prioritized using patient "
        "evidence; illness scripts require patient data linked to distinguishing features of diagnoses; patient "
        "data-to-clinical information requires several findings linked to diagnosis, testing, treatment, epidemiology, "
        "or risk; patient data-to-basic science requires explicit links to foundational science. Working-diagnosis "
        "pathophysiology requires a coherent cause-to-physiologic-change-to-manifestation mechanism; patient-data "
        "pathophysiology requires multiple key findings explicitly explained by mechanism. Transfer requires prior "
        "knowledge visibly applied to deepen current reasoning. A list without those links merits 1 or 2 as appropriate.\n"
        "\nSTRICT DOMAIN AND OVERALL REVIEW\n"
        "A domain receives Yes only when its central educational purpose is sufficiently demonstrated through accurate, "
        "meaningfully connected evidence. Do not assign Yes merely because every criterion has some content, no section "
        "is blank, an average appears acceptable, or terminology is present. Use No when evidence is mainly superficial "
        "or list-like, required relationships are largely absent, central criterion types are only partly demonstrated, "
        "or major criteria are 1 or 2 for substantive reasons.\n"
        "The overall decision is Yes only when the map sufficiently demonstrates knowledge acquisition, integration, "
        "application, and transfer. Before Yes, verify internally at least one clear Integration strength, one clear "
        "Application strength, and evidence beyond isolated fact listing. Use No when central-domain weakness materially "
        "undermines performance, including disconnected facts, absent patient-specific integration, named but unexplained "
        "pathophysiology, unprioritized DDx, patient findings not linked to mechanisms, merely asserted transfer, or "
        "multiple central criteria only partially demonstrated. Do not require catastrophic failure for No. Improvement "
        "feedback may exist for a successful map and does not alone imply No.\n"
        "\nDOMAIN DECISION REFINEMENT\n"
        "A domain decision is holistic: do not determine it from its lowest score alone. Use Yes when the central "
        "educational objective is substantially demonstrated, even with one or more 2s or 3s, minor omissions, "
        "unlabeled arrows, or partially implicit links. Use No only when the domain's central purpose is not "
        "demonstrated—not merely because one criterion is incomplete.\n"
        "For Knowledge Acquisition, basic science, clinical science, and patient-case information are core evidence; "
        "health-system science and determinants of health support the domain but should not alone force No unless "
        "directly relevant to the case or expected map. Do not require policy, provider roles, insurance, quality "
        "improvement, cost, or system organization unless visible or case-relevant. Do not require social, environmental, "
        "behavioral, economic, or structural determinants in every clinically focused map. Limited coverage of either "
        "supporting area is usually a minor limitation, not a domain failure.\n"
        "For Integration, core evidence is a differential, illness scripts, and connections among patient data, clinical "
        "information, and science; one weak link or a partially implicit differential does not alone force No. A differential "
        "may be prioritized by hierarchy, central placement, ordering, stronger links, a working diagnosis, or comparison "
        "of alternatives; a numbered list is not required.\n"
        "For Application, the working diagnosis-to-pathophysiology and patient-data-to-pathophysiology criteria are core. "
        "If one is clear and the other mostly demonstrated, the domain may be Yes. Compare those scores with basic-to-"
        "foundational-science and patient-data-to-basic-science: when explanations show substantial causal understanding, "
        "a missing detail is usually 3 rather than 2 unless a specific visible mechanism is missing.\n"
        "For Transfer, prior basic science, prior clinical concepts, and their use to deepen current reasoning are core. "
        "Deepening understanding includes explaining findings, clarifying mechanisms, supporting the differential or working "
        "diagnosis, and informing management or prognosis; it is not limited to lifestyle or history.\n"
        "A strong map can meet expectations with several 3s, one or two secondary 2s, implicit relationships, or incomplete "
        "coverage. A domain Yes does not require every criterion to be 3 or 4, and an overall Yes does not require every "
        "domain to be flawless. Preserve weak-map discrimination: terminology, density, and lists without meaningful "
        "relationships remain insufficient.\n"
        "Before final JSON, silently check that explanations support scores, 2s are not used merely for minor omissions, "
        "pathophysiology scores do not contradict each other, secondary health-system/DoH limits have not forced No, and "
        "domain and overall decisions reflect the map as a whole. Revise inconsistencies before responding.\n"
        "\nINCOMPLETE VERSUS INADEQUATE EVIDENCE\n"
        "Do not assign 2 merely because a criterion is not fully comprehensive. Use 3 when the required relationship is "
        "substantially demonstrated but lacks detail, omits one explanatory layer, has incomplete labeling, leaves some "
        "connections implicit, or does not cover every possible mechanism. Use 2 only when evidence is meaningfully "
        "partial, the central relationship is weak, the criterion is mostly descriptive, important causal or clinical "
        "connections are absent, or evidence is inconsistent. 'Does not fully explain' usually indicates 3, not 2.\n"
        "Pathophysiology may be demonstrated at an appropriate clinical or mechanistic level. Do not require exhaustive "
        "molecular pathways, cellular signaling, histology, or every intermediate step unless the rubric specifically "
        "requires it. Accurate links from patient finding to mechanism to disease process to diagnosis can earn 3 or 4; "
        "4 requires strong, specific reasoning, not encyclopedic completeness.\n"
        "For patient_data_pathophysiology, assess patient-specific finding-to-mechanism links: symptom, physical finding, "
        "laboratory result, risk factor, or clinical feature to pathophysiologic process. Do not score it primarily from "
        "treatment, epidemiology, diagnostics, or health-system content.\n"
        "Transfer need not be labeled 'previously learned.' It is shown when anatomy, physiology, pharmacology, pathology, "
        "microbiology, diagnostic concepts, or clinical patterns are accurately applied to explain the case, support the "
        "differential, clarify mechanisms, or inform diagnosis, management, or prognosis. Do not lower prior_basic_science "
        "or prior_clinical_concepts merely because the map lacks exhaustive detail after acknowledging their meaningful use.\n"
        "Silently check cross-domain consistency without copying scores: recognized science, illness-script, differential, "
        "and patient-to-diagnosis evidence should support logically consistent Integration, Application, and Transfer scores. "
        "If symptoms, signs, or labs are connected to a working diagnosis, patient_data_pathophysiology is not 2 unless its "
        "mechanism is genuinely missing.\n"
        "Knowledge Acquisition generally may be Yes when basic science, clinical science, and patient-case information are "
        "substantially demonstrated; limited health-system science or DoH alone should not force No unless central to the case. "
        "Integration may be Yes when several meaningful links among patient data, clinical concepts, DDx, illness scripts, "
        "and foundational science are demonstrated; two incomplete links do not automatically defeat strong integration. "
        "Application may be Yes when diagnosis and patient findings are meaningfully tied to mechanism despite missing detail. "
        "Transfer may be Yes when prior science and clinical concepts visibly deepen reasoning.\n"
        "Evidence quality—not evidence perfection—controls the score: weak maps have absent, generic, or unsupported "
        "relationships; adequate maps substantially demonstrate important relationships despite some incompleteness; strong "
        "maps are accurate, specific, organized, and consistently connected. Do not confuse not-perfect with weak.\n"
        "\nCONTRASTIVE EXAMPLES (illustrative only; grade the submitted map independently)\n"
        "Strong: a map links patient symptoms and exam findings to a prioritized differential, connects the working "
        "diagnosis to a coherent pathophysiologic pathway, and ties management to patient-specific findings. Expected: "
        "high Integration/Application scores and overall Yes.\n"
        "Weak: a map lists symptoms, diagnoses, medications, and physiology terms without clearly connecting them, "
        "prioritizing the differential, or explaining findings. Expected: mostly 1 or 2 in Integration/Application and "
        "overall No.\n"
        "Before returning JSON, silently perform an anti-inflation review. For every 3 or 4, verify demonstrated—not "
        "merely mentioned—evidence, visible required relationships, and an evidence-specific explanation; lower superficial "
        "or weakly connected evidence. For every domain Yes and overall Yes, verify reasoning and connections rather than "
        "descriptive lists, including sufficiently developed Integration and Application. Do not output this review.\n\n"
        "OUTPUT CONTRACT\n"
        "Return exactly one valid JSON object. Do not return Markdown, headings, bullets, code fences, "
        "introductory text, trailing commentary, single quotes, comments, trailing commas, NaN, or Infinity. "
        "The first character must be { and the final character must be }. Use double quotes for all keys and "
        "string values. Every required field must be present; do not abbreviate or rename any schema key.\n"
        'The top-level fields "strengths" and "areas_for_improvement" are mandatory. Each must be a JSON '
        "array of 1 to 3 non-empty strings. Do not omit, rename, or return either field as a single string.\n"
        "REQUIRED JSON SCHEMA (placeholders describe types only; replace them with real values):\n"
        + json.dumps(_schema_template(), separators=(",", ":"))
    )


def build_full_retry_prompt(
    map_file: str, reference_materials: list[dict[str, str]] | None = None
) -> str:
    """Compact recovery prompt: exact rubric/schema, without the initial request's calibration essays."""
    reference_context = _compress_reference_materials(reference_materials)
    reference_section = (
        "\nREFERENCE SUMMARY (comparison standard only; never map evidence)\n" + reference_context + "\n"
        if reference_context else ""
    )
    rubric_lines: list[str] = []
    for domain, fields in RETRY_CRITERION_LABELS.items():
        rubric_lines.append(domain.upper())
        for key, label in fields.items():
            rubric_lines.append(f"- {key}: {label}")
    compact_rubric = "\n".join(rubric_lines)
    return (
        "Your previous response was incomplete because it returned only a summary. Return the complete rubric "
        "evaluation this time. Inspect the supplied concept-map image. Do not write introductory text or summary "
        "fields before constructing the rubric domains. The JSON is invalid unless all four domains and all 15 "
        "criterion scores are included.\n\nREQUIRED JSON SCHEMA\n"
        + json.dumps(_schema_template(), separators=(",", ":"))
        + "\n\nTASK\nGrade the student concept map using the exact Spring 2025 Concept Map Feedback Tool below. "
        "Evaluate only visible map content; references define expected content but are not map evidence.\n"
        + reference_section
        + "\nEXACT SPRING 2025 CRITERIA\n" + compact_rubric
        + "\nCOMPACT CALIBRATION\n"
        "Score 4: clearly and substantially demonstrates the criterion with accurate, specific, connected evidence; "
        "perfection is not required. Score 3: mostly demonstrates it with one meaningful limitation. Score 2: some "
        "relevant evidence is present but only partially, superficially, or inconsistently demonstrated. Score 1: "
        "absent, largely unsupported, seriously incorrect, or isolated terms. Presence of terminology alone does not "
        "demonstrate integration, application, or transfer. Do not infer required relationships from proximity, a "
        "diagnosis alone, symptoms beside a diagnosis, arrows without a clear link, or a named mechanism. When the "
        "relationship is uncertain, use the lower score.\n\n"
        "Domain decisions are holistic: a central objective may meet expectations despite one secondary 2 or 3. For "
        "Knowledge Acquisition, basic science, clinical science, and patient-case information are core; limited health-"
        "system science or determinants of health should not alone force No unless case-relevant. For Integration and "
        "Application, require meaningful visible relationships, but one weaker link does not alone force No when the "
        "central reasoning is strong. Transfer includes prior knowledge used to explain findings, mechanisms, diagnosis, "
        "management, or prognosis—not only lifestyle/history. Do not use score 2 for a minor omission when substantial "
        "demonstration warrants 3. Incomplete-but-substantial evidence is 3, not 2: pathophysiology may be clinical or "
        "mechanistic without exhaustive molecular detail. Assess patient_data_pathophysiology through finding-to-mechanism "
        "links, not treatment or epidemiology. Transfer can be inferred from prior science or clinical concepts applied to "
        "the case; it need not be explicitly labeled as previous learning.\n\nMANDATORY COMPLETENESS CHECK\n"
        "Before responding, silently verify knowledge_acquisition, integration, application, and transfer are present; "
        "all 15 criteria have integer scores 1-4 and explanations; all domain decisions, overall_meets_expectations, "
        "strengths (array), areas_for_improvement (array), and grading_notes are present. A summary-only response is "
        "invalid. Return the full JSON object only."
    )


def _vision_messages(prompt: str, image_base64: str) -> list[dict[str, Any]]:
    """NVIDIA NIM OpenAI-compatible multimodal message: text plus a JPEG data URL."""
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{IMAGE_MIME_TYPE};base64,{image_base64}"}},
    ]}]


def _nvidia_payload(
    messages: list[dict[str, Any]], *, response_format: bool = False,
    temperature: float = TEMPERATURE, top_p: float = TOP_P, max_tokens: int = MAX_TOKENS,
    stream: bool = False,
) -> dict[str, Any]:
    payload = {
        "messages": messages,
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _post_nvidia(
    client: dict[str, Any], payload: dict[str, Any], *, stream: bool = False,
    timeout: int | tuple[int, int] = TIMEOUT_SECONDS,
) -> NvidiaChatCompletion:
    endpoint = f"{BASE_URL}/chat/completions"
    started_at = time.monotonic()
    headers = dict(client["headers"])
    if stream:
        headers["Accept"] = "text/event-stream"
    response = client["requests"].post(
        endpoint, headers=headers, json=payload, stream=stream, timeout=timeout
    )
    headers = dict(getattr(response, "headers", {}) or {})
    if stream:
        raw_lines: list[str] = []
        chunks: list[str] = []
        finish_reason: Any = None
        usage: Any = None
        time_to_first_token: float | None = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            raw_lines.append(line)
            if not line.startswith("data:"):
                continue
            event_text = line[5:].strip()
            if event_text == "[DONE]":
                break
            try:
                event = json.loads(event_text)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if time_to_first_token is None:
                        time_to_first_token = round(time.monotonic() - started_at, 3)
                    chunks.append(content)
                finish_reason = choice.get("finish_reason") or finish_reason
            usage = event.get("usage") or usage
        response_text = "\n".join(raw_lines)
        data = {
            "choices": [{"finish_reason": finish_reason, "message": {"content": "".join(chunks)}}],
            "usage": usage or {}, "model": MODEL,
        }
    else:
        response_text = response.text
        try:
            data = response.json()
        except (TypeError, ValueError):
            data = None
        time_to_first_token = None
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
        "streaming_enabled": stream,
        "time_to_first_token_seconds": time_to_first_token,
        "connect_timeout_seconds": timeout[0] if isinstance(timeout, tuple) else timeout,
        "read_timeout_seconds": timeout[1] if isinstance(timeout, tuple) else timeout,
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
        "character }. Every required field must be present. The top-level fields \"strengths\" and "
        "\"areas_for_improvement\" are mandatory arrays of 1 to 3 non-empty strings; do not rename or "
        "return either as a string. If the prior response has a Grading Notes section, convert its positive "
        "observations into strengths and its stated limitations or missing evidence into areas_for_improvement. "
        "You may summarize only evidence and limitations already present; do not invent facts.\n\nPREVIOUS RESPONSE:\n"
        + previous_response
        + "\n\nREQUIRED JSON SCHEMA:\n"
        + json.dumps(_schema_template(), separators=(",", ":"))
        + "\n\nFINAL CHECKLIST BEFORE RESPONDING\n"
        "- All 15 rubric criteria are present.\n"
        "- Every score is an integer from 1 through 4.\n"
        "- All four domain decisions are present.\n"
        "- The final overall decision is present.\n"
        "- The overall explanation is present.\n"
        "- \"strengths\" is present as an array of 1 to 3 strings.\n"
        "- \"areas_for_improvement\" is present as an array of 1 to 3 strings.\n"
        "- No required field is omitted.\n"
        "- Return JSON only."
    )
    return _post_nvidia(
        client,
        _nvidia_payload(
            [{"role": "user", "content": repair_prompt}],
            response_format=response_format,
            temperature=0,
        ),
    )


def request_complete_grading_retry(
    client: Any, retry_prompt: str, image_base64: str, *, response_format: bool
) -> NvidiaChatCompletion:
    """One new multimodal evaluation when the first response omitted substantive rubric grading."""
    return _post_nvidia(
        client,
        _nvidia_payload(
            _vision_messages(retry_prompt, image_base64), response_format=response_format,
            temperature=0, top_p=1, max_tokens=FULL_RETRY_MAX_TOKENS,
            stream=True,
        ),
        stream=True,
        timeout=(CONNECT_TIMEOUT_SECONDS, FULL_RETRY_READ_TIMEOUT_SECONDS),
    )
def _is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return status in {429, 502, 503, 504} or "timeout" in error.__class__.__name__.lower()


def _request_with_retry(
    request: Any, progress_callback: Any | None = None, *, retry_timeouts: bool = True
) -> tuple[Any, dict[str, Any]]:
    try:
        response = request()
        return response, {"request_count": 1, "retry_attempted": False, "http_status": response.transport.get("http_status")}
    except Exception as first_error:
        is_timeout = "timeout" in first_error.__class__.__name__.lower() or "timed out" in str(first_error).lower()
        if not _is_transient(first_error) or (is_timeout and not retry_timeouts):
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


def _lenient_json_object(text: str) -> dict[str, Any] | None:
    """Recover a complete JSON object surrounded by Markdown/prose solely for inspection."""
    cleaned = clean_json_output(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def inspect_grading_completeness(text: str) -> dict[str, Any]:
    """Classify content before selecting a formatter or a full multimodal retry."""
    parsed = _lenient_json_object(text)
    domains_present: list[str] = []
    criteria_present: list[str] = []
    criteria_missing: list[str] = []
    recoverable_scores: dict[str, int] = {}
    all_explanations_present = False
    all_domain_decisions_present = False
    if isinstance(parsed, dict):
        candidate = json.loads(json.dumps(parsed))
        _normalize_scores(candidate)
        recoverable_scores = _score_snapshot(candidate)
        for group, fields in CATEGORY_FIELDS.items():
            section = candidate.get(group)
            if isinstance(section, dict):
                domains_present.append(group)
            for field in fields:
                path = f"{group}.{field}"
                item = section.get(field) if isinstance(section, dict) else None
                if isinstance(item, dict) and path in recoverable_scores:
                    criteria_present.append(path)
                else:
                    criteria_missing.append(path)
        all_explanations_present = all(
            isinstance(candidate.get(group), dict)
            and all(isinstance(candidate[group].get(field), dict)
                    and isinstance(candidate[group][field].get("explanation"), str)
                    for field in fields)
            for group, fields in CATEGORY_FIELDS.items()
        )
        all_domain_decisions_present = all(
            isinstance(candidate.get(group), dict) and "overall_decision" in candidate[group]
            for group in CATEGORY_FIELDS
        )
    domains_missing = [group for group in CATEGORY_FIELDS if group not in domains_present]
    score_count = len(recoverable_scores)
    all_scores_recoverable = score_count == REQUIRED_SCORE_COUNT
    summary_keys = {"map_file", "model", "overall_meets_expectations", "strengths", "areas_for_improvement", "grading_notes"}
    summary_only = isinstance(parsed, dict) and bool(parsed) and set(parsed).issubset(summary_keys)
    complete = (
        not domains_missing and not criteria_missing and all_scores_recoverable
        and all_explanations_present and all_domain_decisions_present
    )
    return {
        "classification": "format_only_failure" if complete else "incomplete_grading_failure",
        "domains_present": domains_present,
        "domains_missing": domains_missing,
        "criteria_present": criteria_present,
        "criteria_missing": criteria_missing,
        "original_score_count": score_count,
        "all_original_scores_recoverable": all_scores_recoverable,
        "summary_only_response": summary_only,
        "parsed": parsed,
    }


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

    validated = parse_model_json(json.dumps(result, separators=(",", ":")), normalize_decisions=True)
    _validate_narrative_fields(validated)
    return validated


def _narrative_field_present(result: dict[str, Any], field: str) -> bool:
    return field in result and isinstance(result.get(field), list)


def _narrative_fields_valid(result: dict[str, Any]) -> bool:
    try:
        _validate_narrative_fields(result)
    except RuntimeError:
        return False
    return True


def _validate_narrative_fields(result: dict[str, Any]) -> None:
    """Llama-only completeness checks; Python never supplies narrative fallback text."""
    for field in ("strengths", "areas_for_improvement"):
        value = result.get(field)
        if not isinstance(value, list):
            raise RuntimeError(f"'{field}' must be a JSON array of 1 to 3 non-empty strings.")
        if not 1 <= len(value) <= 3:
            raise RuntimeError(f"'{field}' must contain 1 to 3 non-empty strings.")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise RuntimeError(f"'{field}' must contain only non-empty strings.")
            stripped = item.strip()
            if stripped.startswith("<") and stripped.endswith(">"):
                raise RuntimeError(f"'{field}' contains a placeholder value.")


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


def _decision_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """Track raw model decisions when JSON was already parseable before repair."""
    decisions = {"overall_meets_expectations": result.get("overall_meets_expectations")}
    for group in CATEGORY_FIELDS:
        section = result.get(group)
        decisions[f"{group}.overall_decision"] = (
            section.get("overall_decision") if isinstance(section, dict) else None
        )
    return decisions


def _decision_debug_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Expose the model's decisions and score distribution without changing its judgment."""
    domain_decisions: dict[str, Any] = {}
    scored_three: list[str] = []
    scored_four: list[str] = []
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        domain_decisions[group] = section.get("overall_decision") if isinstance(section, dict) else None
        if not isinstance(section, dict):
            continue
        for field in fields:
            item = section.get(field)
            score = item.get("score") if isinstance(item, dict) else None
            if score == 3:
                scored_three.append(f"{group}.{field}")
            elif score == 4:
                scored_four.append(f"{group}.{field}")
    overall = result.get("overall_meets_expectations")
    return {
        "domain_decisions": domain_decisions,
        "overall_decision": overall,
        # This is intentionally not inferred by Python; the model's grading_notes is retained verbatim.
        "overall_no_substantial_deficiency_identified": "not independently assessed by Python",
        "overall_decision_explanation": result.get("grading_notes"),
        "criteria_scored_3": scored_three,
        "criteria_scored_4": scored_four,
    }


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
        "strengths_present_initial": False, "areas_for_improvement_present_initial": False,
        "strengths_present_after_repair": False, "areas_for_improvement_present_after_repair": False,
        "narrative_field_repair_required": False, "final_schema_valid": False,
        "initial_response_classification": None, "initial_domains_present": [],
        "initial_domains_missing": [], "initial_criteria_present": [],
        "initial_criteria_missing": [], "initial_original_score_count": 0,
        "initial_all_scores_recoverable": False, "initial_summary_only_response": False,
        "format_repair_eligible": False, "full_multimodal_retry_required": False,
        "full_multimodal_retry_attempted": False, "image_resent_for_full_retry": False,
        # NIM JSON Schema support is not established for this endpoint/model; use json_object only.
        "strict_json_schema_requested": False, "strict_json_schema_supported": None,
        "llama_recovery_version": "incomplete-grading-retry-v2",
        "initial_request": {
            "prompt_character_count": len(prompt), "prompt_token_count": round(len(prompt) / 4),
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P,
            "connect_timeout_seconds": TIMEOUT_SECONDS, "read_timeout_seconds": TIMEOUT_SECONDS,
        },
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
    inspection = inspect_grading_completeness(raw_text)
    debug.update({
        "initial_domains_present": inspection["domains_present"],
        "initial_domains_missing": inspection["domains_missing"],
        "initial_criteria_present": inspection["criteria_present"],
        "initial_criteria_missing": inspection["criteria_missing"],
        "initial_original_score_count": inspection["original_score_count"],
        "initial_all_scores_recoverable": inspection["all_original_scores_recoverable"],
        "initial_summary_only_response": inspection["summary_only_response"],
    })
    initial_parsed: dict[str, Any] | None = None
    initial_scores: dict[str, int] = {}
    initial_decisions: dict[str, Any] = {}
    initial_error_text = ""
    try:
        _, initial_parsed = _parse_json_object(raw_text, attempts)
        debug["initial_json_parse_success"] = True
        debug["strengths_present_initial"] = _narrative_field_present(initial_parsed, "strengths")
        debug["areas_for_improvement_present_initial"] = _narrative_field_present(initial_parsed, "areas_for_improvement")
        debug["narrative_field_repair_required"] = not _narrative_fields_valid(initial_parsed)
        initial_normalizations = _normalize_scores(initial_parsed)
        initial_scores = _score_snapshot(initial_parsed)
        initial_decisions = _decision_snapshot(initial_parsed)
        validated = _validate_existing_schema(initial_parsed)
        debug["initial_schema_validation_success"] = True
        debug["initial_response_classification"] = "valid_complete_grading"
        parsed_path = Path(f"{debug_prefix}_grading_parsed.json")
        parsed_path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
        debug.update({
            "score_normalizations": initial_normalizations, "parsed_path": str(parsed_path),
            **_decision_debug_metadata(validated),
            "http_status": diagnostics["http_status"], "finish_reason": diagnostics["finish_reason"],
            "usage": diagnostics["usage"], "response_content_length": len(raw_text),
            "final_result_source": "initial", "duration_seconds": round(time.monotonic() - started_at, 3),
            "final_schema_valid": True,
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

    if inspection["classification"] == "incomplete_grading_failure":
        debug.update({
            "initial_response_classification": "incomplete_grading_failure",
            "full_multimodal_retry_required": True,
            "full_multimodal_retry_attempted": True,
            "image_resent_for_full_retry": True,
            # Compatibility field: this is a grading retry, never a format repair.
            "retry_attempted": True,
        })
        if progress_callback:
            progress_callback("Llama returned an incomplete evaluation. Running one complete grading retry...")
        full_retry_prompt = build_full_retry_prompt(map_file, reference_materials)
        debug["full_retry_request"] = {
            "prompt_character_count": len(full_retry_prompt), "prompt_token_count": round(len(full_retry_prompt) / 4),
            "max_tokens": FULL_RETRY_MAX_TOKENS, "temperature": 0, "top_p": 1,
            "connect_timeout_seconds": CONNECT_TIMEOUT_SECONDS,
            "read_timeout_seconds": FULL_RETRY_READ_TIMEOUT_SECONDS,
            "streaming_enabled": True,
            "image_width": image_info["width"], "image_height": image_info["height"],
            "image_bytes": image_info["bytes"], "jpeg_quality": image_info.get("jpeg_quality"),
            "resize_applied": image_info.get("render_matrix"),
        }
        try:
            retry_response, full_retry_debug = _request_with_retry(
                lambda: request_complete_grading_retry(
                    client, full_retry_prompt, image_base64,
                    response_format=debug.get("response_format_supported") is True,
                ),
                progress_callback,
                retry_timeouts=False,
            )
            full_retry_attempts = {"full_multimodal_retry": _response_dump(retry_response)}
            retry_text = response_text(retry_response, full_retry_attempts)
            retry_diagnostics = _response_diagnostics(retry_response)
            retry_raw_path = Path(f"{debug_prefix}_full_retry_raw.txt")
            retry_raw_path.write_text(retry_text, encoding="utf-8")
            debug.update({
                "full_multimodal_retry_request_count": full_retry_debug["request_count"],
                "full_multimodal_retry_http_status": retry_diagnostics["http_status"],
                "full_multimodal_retry_finish_reason": retry_diagnostics["finish_reason"],
                "full_retry_http_status": retry_diagnostics["http_status"],
                "full_retry_finish_reason": retry_diagnostics["finish_reason"],
                "full_multimodal_retry_content_length": len(retry_text),
                "full_multimodal_retry_response": retry_diagnostics,
                "full_multimodal_retry_raw_path": str(retry_raw_path),
                "full_multimodal_retry_max_tokens": FULL_RETRY_MAX_TOKENS,
                "full_retry_prompt_character_count": len(full_retry_prompt),
                "full_retry_prompt_estimated_tokens": round(len(full_retry_prompt) / 4),
                "full_retry_max_tokens": FULL_RETRY_MAX_TOKENS,
                "full_retry_temperature": 0, "full_retry_top_p": 1,
                "full_retry_connect_timeout_seconds": CONNECT_TIMEOUT_SECONDS,
                "full_retry_read_timeout_seconds": FULL_RETRY_READ_TIMEOUT_SECONDS,
                "full_retry_streaming_enabled": True,
                "full_retry_time_to_first_token_seconds": retry_response.transport.get("time_to_first_token_seconds"),
                "full_retry_duration_seconds": retry_response.transport.get("elapsed_request_seconds"),
                "full_retry_response_received": True,
                "full_retry_response_length": len(retry_text),
            })
            _, retried = _parse_json_object(retry_text, full_retry_attempts)
            retry_inspection = inspect_grading_completeness(retry_text)
            if retry_inspection["classification"] != "format_only_failure":
                raise RuntimeError("The complete grading retry still omitted required rubric domains, criteria, or scores.")
            retry_normalizations = _normalize_scores(retried)
            validated = _validate_existing_schema(retried)
            parsed_path = Path(f"{debug_prefix}_grading_parsed.json")
            parsed_path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
            debug.update({
                "full_multimodal_retry_schema_validation_success": True,
                "full_retry_schema_validation_success": True,
                "full_multimodal_retry_score_normalizations": retry_normalizations,
                "parsed_path": str(parsed_path), **_decision_debug_metadata(validated),
                "final_result_source": "full_multimodal_retry", "final_schema_valid": True,
                "duration_seconds": round(time.monotonic() - started_at, 3),
            })
            debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
            return {"model": MODEL, "provider": PROVIDER, "raw_text": retry_text,
                    "cleaned_text": json.dumps(validated, separators=(",", ":")), "response": retry_response,
                    "prompt": prompt, "prompt_path": prompt_path, "image_path": image_path,
                    "raw_path": retry_raw_path, "debug": {**debug, "debug_path": str(debug_path)}}
        except Exception as full_retry_error:
            error_text = str(full_retry_error).lower()
            if "timed out" in error_text or "timeout" in error_text:
                failure_type = "timeout"
                message = "Llama returned an incomplete evaluation. The complete grading retry timed out before NVIDIA returned a response."
            elif isinstance(full_retry_error, NvidiaHttpError):
                failure_type = "http_error"
                message = "Llama returned an incomplete evaluation. NVIDIA returned an error during the complete grading retry."
            elif "schema" in error_text or "required" in error_text:
                failure_type = "schema_error"
                message = "Llama returned an incomplete evaluation. The complete grading retry returned JSON that did not match the required grading schema."
            else:
                failure_type = "parse_error"
                message = "Llama returned an incomplete evaluation. The complete grading retry returned invalid JSON."
            debug.update({
                "full_multimodal_retry_error": str(full_retry_error),
                "full_multimodal_retry_attempts": getattr(full_retry_error, "attempts", None),
                "full_retry_failure_type": failure_type, "full_retry_response_received": False,
                "duration_seconds": round(time.monotonic() - started_at, 3), "final_result_source": "failure",
            })
            debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
            raise MalformedLlamaVisionJsonError(
                debug,
                message,
            ) from full_retry_error

    # A complete but malformed/invalid response gets one formatter request only.  It has no image,
    # reference context, or rubric, so it cannot regrade the map.
    recoverable_initial = inspection.get("parsed")
    if isinstance(recoverable_initial, dict):
        recoverable_initial = json.loads(json.dumps(recoverable_initial))
        _normalize_scores(recoverable_initial)
        initial_scores = _score_snapshot(recoverable_initial)
        initial_decisions = _decision_snapshot(recoverable_initial)
    debug["initial_response_classification"] = "format_only_failure"
    debug["format_repair_eligible"] = True
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
        debug["strengths_present_after_repair"] = _narrative_field_present(repaired, "strengths")
        debug["areas_for_improvement_present_after_repair"] = _narrative_field_present(repaired, "areas_for_improvement")
        repair_normalizations = _normalize_scores(repaired)
        repaired_scores = _score_snapshot(repaired)
        if len(initial_scores) != REQUIRED_SCORE_COUNT or repaired_scores != initial_scores:
            raise RuntimeError("The JSON formatter omitted, added, or changed one or more model-generated scores.")
        if _decision_snapshot(repaired) != initial_decisions:
            raise RuntimeError("The JSON formatter changed one or more model-generated Yes/No decisions.")
        validated = _validate_existing_schema(repaired)
        debug["repair_schema_validation_success"] = True
        parsed_path = Path(f"{debug_prefix}_grading_parsed.json")
        parsed_path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
        debug.update({
            "repair_score_normalizations": repair_normalizations, "parsed_path": str(parsed_path),
            **_decision_debug_metadata(validated),
            "final_result_source": "repair", "duration_seconds": round(time.monotonic() - started_at, 3),
            "final_schema_valid": True,
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
