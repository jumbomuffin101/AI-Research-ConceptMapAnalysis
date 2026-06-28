"""Upload-driven grading pipeline used by the Streamlit demo.

The command-line graders remain independent. This module mirrors their proven
PDF rendering and OpenRouter request flow while accepting any uploaded PDF.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

import fitz
from dotenv import load_dotenv
from openai import OpenAI

from grading import grade_nemotron, grade_qwen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = PROJECT_ROOT / "rubric" / "concept_map_rubric.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "web_demo"
DEBUG_DIR = OUTPUT_DIR / "debug"

GRADER_MODULES = {
    "Gemma": grade_qwen,
    "Nemotron": grade_nemotron,
}

MODEL_SELECTION_ALIASES: dict[str, str] = {}

MODEL_CONFIGS = {
    "Gemma": {
        "model_id": grade_qwen.MODEL,
        "max_tokens": 2000,
    },
    "Nemotron": {
        "model_id": grade_nemotron.MODEL,
        "max_tokens": 1500,
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

DOMAIN_OVERALL_QUESTIONS = {
    "knowledge_acquisition": (
        "Does the student's map include key knowledge from the case and content "
        "learned during this unit?"
    ),
    "integration": "Did the learner connect key knowledge accurately & comprehensively?",
    "application": "Did the learner explain key clinical data with relevant basic science?",
    "transfer": "Did the learner use previously learned content to deepen understanding?",
}

FORBIDDEN_DECISION_TEXT = (
    "Partial",
    "Partially Meets",
    "Borderline",
    "Maybe",
    "Almost",
    "Meets some expectations",
)


class GradingError(RuntimeError):
    """A user-facing failure while grading a concept map."""


class InvalidPDFError(GradingError):
    """The uploaded file cannot be read as a PDF."""


class ModelResponseError(GradingError):
    """A model returned no usable grading response."""

    def __init__(self, message: str, raw_response: Any | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class MalformedResultError(GradingError):
    """A model response is not valid grading JSON."""


@dataclass(frozen=True)
class EvaluationResult:
    """One model's parsed result and persisted output location."""

    model_name: str
    model_id: str
    data: dict[str, Any]
    output_path: Path


@dataclass(frozen=True)
class EvaluationFailure:
    """One model's failed result and persisted debug location."""

    model_name: str
    model_id: str
    error_message: str
    debug_path: Path


EvaluationOutcome = EvaluationResult | EvaluationFailure


def selected_model_names(selection: str) -> list[str]:
    """Translate the UI selection into model registry keys."""
    if selection == "Both":
        return ["Gemma", "Nemotron"]
    selection = MODEL_SELECTION_ALIASES.get(selection, selection)
    if selection not in MODEL_CONFIGS:
        raise GradingError(f"Unknown model selection: {selection}")
    return [selection]


def render_pdf_image(pdf_path: Path, model_name: str) -> str:
    """Render the uploaded PDF as a deployment-safe base64 PNG."""
    _ = model_name
    try:
        with fitz.open(pdf_path) as document:
            if document.page_count < 1:
                raise InvalidPDFError("The uploaded PDF has no pages.")
            page = document[0]
            # Lower resolution for Streamlit/OpenRouter token compatibility
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            image_bytes = pixmap.tobytes("png")
            return base64.b64encode(image_bytes).decode("utf-8")
    except InvalidPDFError:
        raise
    except (
        fitz.FileDataError,
        fitz.EmptyFileError,
        RuntimeError,
        ValueError,
        IndexError,
    ) as exc:
        raise InvalidPDFError("The uploaded file is not a valid, readable PDF.") from exc


def load_summative_rubric() -> dict[str, Any]:
    """Load the Spring 2025 summative rubric used by grading prompts."""
    try:
        rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GradingError(f"Rubric not found at {RUBRIC_PATH}.") from exc
    except json.JSONDecodeError as exc:
        raise GradingError("The rubric file contains invalid JSON.") from exc

    return {
        group: rubric[group]
        for group in CATEGORY_FIELDS
        if isinstance(rubric.get(group), dict)
    }


def build_spring_schema(map_file: str, model_id: str) -> dict[str, Any]:
    """Build the Spring 2025 summative grading JSON shape."""
    schema: dict[str, Any] = {"map_file": map_file, "model": model_id}
    for group, fields in CATEGORY_FIELDS.items():
        schema[group] = {
            field: {
                "score": 1,
                "explanation": "",
                "evidence_from_map": [],
            }
            for field in fields
        }
        schema[group]["overall_decision"] = "No"
        schema[group]["if_no_explanation"] = ""
    schema["overall_meets_expectations"] = "No"
    schema["strengths"] = ["", ""]
    schema["areas_for_improvement"] = ["", ""]
    schema["grading_notes"] = ""
    return schema


def compact_spring_rubric_text(rubric: dict[str, Any]) -> str:
    """Represent every Spring 2025 rubric criterion without JSON overhead."""
    _ = rubric
    return """Shared scale: L=little/irrelevant; P=partly relevant and/or too general; R=relevant and mostly synthesized; D=synthesized and detailed; C1=inaccurate/illogical connections; C2=mostly accurate but simplistic/errors; C3=accurate logical flow; C4=accurate logical comprehensive.
knowledge_acquisition overall: Does the student's map include key knowledge from the case and content learned during this unit?
knowledge_acquisition.basic_science: 1=L knowledge; 2=P knowledge; 3=R knowledge; 4=D knowledge from each session.
knowledge_acquisition.health_system_science: 1=L knowledge; 2=P knowledge; 3=R knowledge; 4=D knowledge from multiple sessions.
knowledge_acquisition.clinical_science: 1=L knowledge; 2=P knowledge; 3=R knowledge; 4=relevant, synthesized, detailed knowledge from each session.
knowledge_acquisition.patient_case_information: 1=little/irrelevant information; 2=partly relevant with limited synthesis; 3=relevant mostly synthesized patient data; 4=synthesized, relevant, comprehensive patient data.
knowledge_acquisition.determinants_of_health: 1=DoH absent; 2=at least one DoH but not patient-specific and/or not clinically relevant; 3=multiple DoH across map with clear impact on condition and/or care; 4=comprehensive DoH across map with clear impact on condition, care, and/or prognosis.
integration overall: Did the learner connect key knowledge accurately & comprehensively?
integration.prioritized_differential_diagnosis: 1=DDx absent and/or mostly incorrect; 2=too narrow or not accurately connected to patient data; 3=focused and relevant to patient; 4=focused, relevant, correctly prioritized.
integration.illness_scripts: 1=insufficient data for illness scripts; 2=incorrect/incomplete; 3=accurate patient-data connections; 4=accurate prioritized patient-data connections for multiple diagnoses.
integration.basic_to_foundational_science: 1=C1; 2=C2; 3=C3 from unit basic science to molecular/cellular disease basis; 4=C4 including anatomy, histology, biochemistry, genetics, physiology, and/or pharmacology.
integration.patient_data_to_clinical_information: 1=C1; 2=C2; 3=C3 from patient data to clinical information; 4=C4 including epidemiology, symptoms, signs, diagnostics, treatments, and patient-specific risk factors.
integration.patient_data_to_basic_science: 1=C1; 2=C2; 3=C3 from patient data to molecular/cellular disease basis; 4=C4 including anatomy, histology, biochemistry, genetics, physiology, and/or pharmacology.
application overall: Did the learner explain key clinical data with relevant basic science?
application.working_diagnosis_pathophysiology: 1=pathophysiology connections absent/unclear; 2=present but inaccurate and/or too simplistic; 3=flow of concepts explains pathophysiology; 4=flow explains pathophysiology and includes basic, clinical, health-system sciences.
application.patient_data_pathophysiology: 1=pathophysiology connections absent/unclear; 2=present but inaccurate and/or too simplistic; 3=flow of concepts explains pathophysiology; 4=flow explains pathophysiology of multiple patient-data components including symptoms, signs, findings, and/or care plan.
transfer overall: Did the learner use previously learned content to deepen understanding?
transfer.prior_basic_science: 1=L knowledge; 2=P knowledge; 3=R knowledge; 4=D knowledge.
transfer.prior_clinical_concepts: 1=L knowledge; 2=P knowledge; 3=R knowledge; 4=D knowledge.
transfer.deepens_understanding: 1=L knowledge; 2=P knowledge; 3=R knowledge; 4=D knowledge connecting patient data and basic science."""


def compact_output_contract() -> str:
    """Describe required JSON fields compactly for Nemotron."""
    domains = "; ".join(
        f"{group}=[{','.join(fields)}]+overall_decision+if_no_explanation"
        for group, fields in CATEGORY_FIELDS.items()
    )
    return (
        "Top keys: map_file,model,knowledge_acquisition,integration,application,"
        "transfer,overall_meets_expectations,strengths,areas_for_improvement,"
        f"grading_notes. Domains: {domains}. "
        "Every criterion object has exactly score,explanation,evidence_from_map."
    )


def build_web_prompt(map_file: str, model_id: str) -> str:
    """Build the shorter Streamlit/OpenRouter-compatible grading prompt."""
    schema = build_spring_schema(map_file, model_id)
    rubric = load_summative_rubric()

    return f"""Use the Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities exactly.
Do not invent additional grading criteria.

Rubric:
{json.dumps(rubric, indent=2)}

Global rules:
- Every criterion score must be an integer 1, 2, 3, or 4 only.
- Every domain overall_decision must be exactly "Yes" or "No".
- overall_meets_expectations must be exactly "Yes" or "No".
- Do not output Partial, Partially Meets, Borderline, Maybe, score 0, score 5, decimal scores, or any score outside 1-4.
- If evidence is missing, write "No clear evidence found in the concept map."
- Do not hallucinate evidence not visible in the concept map.

For every scored category, return only:
- score: integer 1-4
- explanation: one short explanation
- evidence_from_map: short strings copied or paraphrased from visible map content

Each domain must include:
- overall_decision: "Yes" or "No"
- if_no_explanation: required when overall_decision is "No"; otherwise empty string

Keep all JSON string values short. Do not write paragraphs.
Return JSON only. Do not include markdown or text outside JSON.
Use this exact JSON structure:
{json.dumps(schema, indent=2)}
"""


def build_nemotron_web_prompt(map_file: str, model_id: str) -> str:
    """Build Nemotron's compact web prompt for more reliable JSON output."""
    rubric = load_summative_rubric()

    return (
        "Spring 2025 SUMMATIVE concept map grading. Use only visible map evidence.\n"
        "Rules: score each criterion with integer 1-4 only; never use 0,5,decimals,"
        "Partial,Partially Meets,Borderline,Maybe. Domain overall_decision and "
        "overall_meets_expectations must be exactly Yes or No. Each criterion needs "
        "score, explanation, evidence_from_map. If evidence is missing, write "
        "\"No clear evidence found in the concept map.\" Do not hallucinate evidence. "
        "If a domain is No, fill if_no_explanation. Return only valid minified JSON.\n"
        f"Rubric:\n{compact_spring_rubric_text(rubric)}\n"
        f"JSON contract: {compact_output_contract()} "
        f'Use map_file="{map_file}" and model="{model_id}".'
    )


def build_nemotron_domain_prompt(domain: str) -> str:
    """Build one Spring 2025 domain request with its complete rubric criteria."""
    rubric = load_summative_rubric()[domain]
    fields = CATEGORY_FIELDS[domain]
    domain_schema = {
        field: {"score": 1, "explanation": "", "evidence_from_map": []}
        for field in fields
    }
    domain_schema.update({"overall_decision": "No", "if_no_explanation": ""})
    return f"""Grade only the {domain} domain using the Spring 2025 SUMMATIVE rubric below.
Use only visible concept-map evidence. Apply every criterion and its 1-4 definitions exactly.
Every score must be an integer 1-4. overall_decision must be exactly "Yes" or "No".
Each criterion needs a brief explanation and 1-2 short evidence items. Do not invent evidence.
If evidence is absent, use "No clear evidence found in the concept map."
If overall_decision is "No", provide if_no_explanation; otherwise use an empty string.
Return only valid minified JSON for this domain, with no markdown or prose.
Rubric: {json.dumps(rubric, separators=(',', ':'))}
Required JSON: {json.dumps(domain_schema, separators=(',', ':'))}"""


def build_nemotron_summary_prompt(domains: dict[str, Any]) -> str:
    """Build the final decision request from the four completed domain outputs."""
    summary_schema = {
        "overall_meets_expectations": "No",
        "strengths": [],
        "areas_for_improvement": [],
        "grading_notes": "",
    }
    return f"""Summarize these four Spring 2025 domain grading results.
Base the summary only on the supplied domain JSON. Do not rescore or change any domain.
overall_meets_expectations must be exactly "Yes" or "No".
Return up to two short strengths, up to two short areas_for_improvement, and one brief grading_notes string.
Return only valid minified JSON with no markdown or prose.
Domains: {json.dumps(domains, separators=(',', ':'))}
Required JSON: {json.dumps(summary_schema, separators=(',', ':'))}"""


def build_model_prompt(model_name: str, map_file: str, model_id: str) -> str:
    """Build the web prompt for a model without changing CLI prompts."""
    if model_name == "Nemotron":
        # Nemotron is an experimental secondary grader and may require JSON cleanup.
        return build_nemotron_web_prompt(map_file, model_id)
    return build_web_prompt(map_file, model_id)


def _strip_json_fences(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s*```\s*$", "", text)


def _extract_first_complete_json_object(text: str) -> str | None:
    """Return the first balanced JSON object substring, if one exists."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_repairable_json_object(text: str) -> str | None:
    """Extract JSON and append missing closers only when structurally obvious."""
    start = text.find("{")
    if start < 0:
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    last_index = start

    for index in range(start, len(text)):
        char = text[index]
        last_index = index
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]

    if in_string or escaped or not stack:
        return None

    candidate = text[start : last_index + 1].rstrip()
    candidate = re.sub(r",\s*$", "", candidate)
    return candidate + "".join(reversed(stack))


def _load_json_with_repair(raw_text: str) -> dict[str, Any]:
    """Parse JSON, then fall back to the first complete object if needed."""
    text = _strip_json_fences(raw_text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as first_exc:
        candidate = _extract_first_complete_json_object(text)
        if candidate is None:
            candidate = _extract_repairable_json_object(text)
        if candidate is None:
            raise MalformedResultError(
                "The model returned malformed JSON "
                f"({first_exc.msg}, line {first_exc.lineno})."
            ) from first_exc
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError as second_exc:
            raise MalformedResultError(
                "The model returned malformed JSON "
                f"({second_exc.msg}, line {second_exc.lineno})."
            ) from second_exc

    if not isinstance(result, dict):
        raise MalformedResultError("The model result must be a JSON object.")
    return result


def _contains_forbidden_decision_text(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_forbidden_decision_text(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_decision_text(item) for item in value)
    if not isinstance(value, str):
        return False
    return any(
        re.search(rf"\b{re.escape(term)}\b", value, flags=re.IGNORECASE)
        for term in FORBIDDEN_DECISION_TEXT
    )


def _require_yes_no(value: Any, field_path: str) -> None:
    if value not in {"Yes", "No"}:
        raise MalformedResultError(f"'{field_path}' must be exactly 'Yes' or 'No'.")


def _normalize_yes_no(value: Any) -> str:
    return value if value in {"Yes", "No"} else "No"


def _normalize_score(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return max(1, min(4, int(round(value))))
    if isinstance(value, str):
        try:
            return max(1, min(4, int(round(float(value.strip())))))
        except ValueError:
            return 1
    return 1


def normalize_nemotron_result(result: dict[str, Any]) -> dict[str, Any]:
    """Fill conservative defaults for incomplete Nemotron JSON outputs."""
    normalized = dict(result)
    changed = False

    if normalized.get("overall_meets_expectations") not in {"Yes", "No"}:
        normalized["overall_meets_expectations"] = "No"
        changed = True

    for field, default in (
        ("strengths", []),
        ("areas_for_improvement", []),
        ("grading_notes", ""),
    ):
        if field not in normalized:
            normalized[field] = default
            changed = True

    for group, fields in CATEGORY_FIELDS.items():
        section = normalized.get(group)
        if not isinstance(section, dict):
            section = {}
            normalized[group] = section
            changed = True

        decision = section.get("overall_decision")
        normalized_decision = _normalize_yes_no(decision)
        if decision != normalized_decision:
            section["overall_decision"] = normalized_decision
            changed = True

        if (
            section["overall_decision"] == "No"
            and not str(section.get("if_no_explanation", "")).strip()
        ):
            section["if_no_explanation"] = (
                "Model did not provide a domain-level explanation."
            )
            changed = True
        elif "if_no_explanation" not in section:
            section["if_no_explanation"] = ""
            changed = True

        for criterion in fields:
            item = section.get(criterion)
            if not isinstance(item, dict):
                section[criterion] = {
                    "score": 1,
                    "explanation": "No valid model output provided for this criterion.",
                    "evidence_from_map": [
                        "No clear evidence found in the concept map."
                    ],
                }
                changed = True
                continue

            score = item.get("score")
            normalized_score = _normalize_score(score)
            if score != normalized_score:
                item["score"] = normalized_score
                changed = True

            if not isinstance(item.get("explanation"), str) or not item.get(
                "explanation", ""
            ).strip():
                item["explanation"] = (
                    "No valid model output provided for this criterion."
                )
                changed = True

            evidence = item.get("evidence_from_map")
            if not isinstance(evidence, list) or not evidence:
                item["evidence_from_map"] = [
                    "No clear evidence found in the concept map."
                ]
                changed = True

    if changed:
        normalized["grading_notes"] = (
            "Model output was normalized because required fields were missing."
        )

    return normalized


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Extract, parse, repair when possible, and validate grading JSON."""
    if not raw_text or not raw_text.strip():
        raise ModelResponseError("The model returned an empty response.")

    if "{" not in raw_text:
        raise MalformedResultError("The model response did not contain a JSON object.")

    result = _load_json_with_repair(raw_text)
    if _contains_forbidden_decision_text(result):
        raise MalformedResultError(
            "The model result contains a forbidden non-binary decision label."
        )

    missing = [
        key
        for key in (*CATEGORY_FIELDS.keys(), "overall_meets_expectations")
        if key not in result
    ]
    if missing:
        raise MalformedResultError(
            "The model result is missing required fields: " + ", ".join(missing)
        )

    _require_yes_no(
        result.get("overall_meets_expectations"),
        "overall_meets_expectations",
    )

    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            raise MalformedResultError(f"'{group}' must be a JSON object.")
        _require_yes_no(section.get("overall_decision"), f"{group}.overall_decision")
        if (
            section.get("overall_decision") == "No"
            and not str(section.get("if_no_explanation", "")).strip()
        ):
            raise MalformedResultError(
                f"'{group}.if_no_explanation' is required when overall_decision is 'No'."
            )
        for field in fields:
            item = section.get(field)
            score = item.get("score") if isinstance(item, dict) else None
            if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 4:
                raise MalformedResultError(
                    f"'{group}.{field}.score' must be an integer from 1 to 4."
                )
            if not isinstance(item.get("explanation"), str):
                raise MalformedResultError(
                    f"'{group}.{field}.explanation' must be a string."
                )
    return result


def _response_to_debug_text(response: Any) -> str:
    """Convert an SDK response object into a debug-safe text payload."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    for method_name in ("model_dump_json", "json"):
        method = getattr(response, method_name, None)
        if callable(method):
            try:
                return method(indent=2)
            except TypeError:
                try:
                    return method()
                except Exception:
                    pass
            except Exception:
                pass
    return str(response)


def _save_failed_response(
    *,
    timestamp: str,
    run_id: str,
    file_stem: str,
    model_name: str,
    model_id: str,
    error_message: str,
    raw_response: Any | None,
) -> Path:
    """Persist failed model content for later debugging."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    debug_path = DEBUG_DIR / (
        f"{timestamp}_{run_id}_{file_stem}_{model_name.lower()}_failure.json"
    )
    payload = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_id": model_id,
        "error_message": error_message,
        "raw_response": _response_to_debug_text(raw_response),
    }
    debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return debug_path


def _ensure_api_key() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise GradingError(
            "OPENROUTER_API_KEY is missing. Add it to the environment or project .env file."
        )


def _create_client(*, disable_sdk_retries: bool = False) -> OpenAI:
    _ensure_api_key()
    options: dict[str, Any] = {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 300,
    }
    if disable_sdk_retries:
        options["max_retries"] = 0
    return OpenAI(
        **options
    )


def _is_input_limit_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        phrase in lowered
        for phrase in (
            "prompt tokens limit exceeded",
            "input token",
            "context length",
            "maximum context",
            "token limit",
        )
    )


def _request_model(
    model_name: str,
    prompt: str,
    image: str,
    max_tokens: int | None = None,
    request_timeout: float | None = None,
) -> tuple[str, str]:
    config = MODEL_CONFIGS[model_name]
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image}"},
        },
    ]

    try:
        client = _create_client(disable_sdk_retries=request_timeout is not None)
        request_options: dict[str, Any] = {
            "model": config["model_id"],
            "max_tokens": max_tokens or config["max_tokens"],
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        if request_timeout is not None:
            request_options["timeout"] = request_timeout
        response = client.chat.completions.create(
            **request_options
        )
    except Exception as exc:
        message = str(exc)
        if _is_input_limit_error(message):
            message = (
                "Input is too large for the current OpenRouter model limit. "
                "Try a smaller PDF/image or use the local CLI pipeline."
            )
        raise ModelResponseError(
            f"{model_name} API request failed: {message}",
            raw_response=repr(exc),
        ) from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise ModelResponseError(
            f"{model_name} returned no response choices.",
            raw_response=response,
        )
    try:
        text = choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ModelResponseError(
            f"{model_name} returned a malformed API response.",
            raw_response=response,
        ) from exc

    if not isinstance(text, str) or not text.strip():
        raise ModelResponseError(
            f"{model_name} returned no usable content.",
            raw_response=response,
        )
    return config["model_id"], text


def _validate_nemotron_domain(domain: str, data: dict[str, Any]) -> None:
    """Reject incomplete domain output so the individual step can be retried."""
    missing = [field for field in CATEGORY_FIELDS[domain] if field not in data]
    missing.extend(
        field
        for field in ("overall_decision", "if_no_explanation")
        if field not in data
    )
    if missing:
        raise MalformedResultError(
            f"{domain} output is missing required fields: {', '.join(missing)}"
        )
    _require_yes_no(data.get("overall_decision"), f"{domain}.overall_decision")
    if (
        data.get("overall_decision") == "No"
        and not str(data.get("if_no_explanation", "")).strip()
    ):
        raise MalformedResultError(
            f"'{domain}.if_no_explanation' is required for a No decision."
        )
    for field in CATEGORY_FIELDS[domain]:
        item = data.get(field)
        if not isinstance(item, dict):
            raise MalformedResultError(f"'{domain}.{field}' must be an object.")
        score = item.get("score")
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 4:
            raise MalformedResultError(
                f"'{domain}.{field}.score' must be an integer from 1 to 4."
            )
        if not isinstance(item.get("explanation"), str):
            raise MalformedResultError(f"'{domain}.{field}.explanation' is required.")
        if not isinstance(item.get("evidence_from_map"), list):
            raise MalformedResultError(
                f"'{domain}.{field}.evidence_from_map' must be a list."
            )


def _failed_nemotron_domain(domain: str) -> dict[str, Any]:
    """Return the schema-safe conservative fallback for a failed domain step."""
    section = {
        field: {
            "score": 1,
            "explanation": "No valid model output provided for this criterion.",
            "evidence_from_map": ["No clear evidence found in the concept map."],
        }
        for field in CATEGORY_FIELDS[domain]
    }
    section.update(
        {
            "overall_decision": "No",
            "if_no_explanation": f"Nemotron failed to generate the {domain} domain.",
        }
    )
    return section


def _save_nemotron_step_debug(step: str, attempt: int, error: Exception) -> None:
    """Save each failed sequential attempt without affecting the final UI result."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_step = re.sub(r"[^A-Za-z0-9_-]+", "_", step)
    path = DEBUG_DIR / f"{timestamp}_nemotron_{safe_step}_attempt{attempt}.txt"
    raw = getattr(error, "raw_response", None)
    path.write_text(
        f"error={error}\n\n{_response_to_debug_text(raw)}", encoding="utf-8"
    )


def _request_nemotron_json_step(
    step: str,
    prompt: str,
    image: str,
    max_tokens: int,
    deadline: float,
) -> dict[str, Any]:
    """Run and validate one compact Nemotron call, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, 3):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelResponseError("Nemotron's four-minute run limit was reached.")
        raw_text: str | None = None
        try:
            _, raw_text = _request_model(
                "Nemotron",
                prompt,
                image,
                max_tokens=max_tokens,
                request_timeout=min(60.0, remaining),
            )
            data = _load_json_with_repair(raw_text)
            if step in CATEGORY_FIELDS:
                _validate_nemotron_domain(step, data)
            else:
                required = {
                    "overall_meets_expectations",
                    "strengths",
                    "areas_for_improvement",
                    "grading_notes",
                }
                missing = required.difference(data)
                if missing:
                    raise MalformedResultError(
                        "Summary output is missing required fields: "
                        + ", ".join(sorted(missing))
                    )
                _require_yes_no(
                    data.get("overall_meets_expectations"),
                    "overall_meets_expectations",
                )
                if not isinstance(data.get("strengths"), list):
                    raise MalformedResultError("'strengths' must be a list.")
                if not isinstance(data.get("areas_for_improvement"), list):
                    raise MalformedResultError(
                        "'areas_for_improvement' must be a list."
                    )
                if not isinstance(data.get("grading_notes"), str):
                    raise MalformedResultError("'grading_notes' must be a string.")
            return data
        except (ModelResponseError, MalformedResultError) as exc:
            if getattr(exc, "raw_response", None) is None and raw_text is not None:
                exc.raw_response = raw_text
            last_error = exc
            _save_nemotron_step_debug(step, attempt, exc)
            if attempt == 1:
                remaining = deadline - time.monotonic()
                if remaining <= 3:
                    break
                time.sleep(3)
    raise ModelResponseError(
        f"Nemotron {step} step failed after 2 attempts: {last_error}",
        raw_response=getattr(last_error, "raw_response", None),
    )


def _local_nemotron_summary(
    domains: dict[str, Any], note: str
) -> dict[str, Any]:
    """Create a deterministic summary when the remote summary is unavailable."""
    overall = (
        "No"
        if any(section.get("overall_decision") == "No" for section in domains.values())
        else "Yes"
    )
    return {
        "overall_meets_expectations": overall,
        "strengths": [],
        "areas_for_improvement": [
            "Review domains with low scores or missing evidence."
        ],
        "grading_notes": note,
    }


def _run_nemotron_sequential(
    image: str,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate the full grading object through four domains and one summary."""
    deadline = time.monotonic() + 240
    domains: dict[str, Any] = {}
    failed_domains: list[str] = []
    progress_labels = {
        "knowledge_acquisition": "Knowledge Acquisition",
        "integration": "Integration",
        "application": "Application",
        "transfer": "Transfer",
    }
    for domain in CATEGORY_FIELDS:
        if progress_callback:
            progress_callback(f"Nemotron: grading {progress_labels[domain]}...")
        if time.monotonic() >= deadline:
            domains[domain] = _failed_nemotron_domain(domain)
            failed_domains.append(domain)
            continue
        try:
            domains[domain] = _request_nemotron_json_step(
                domain,
                build_nemotron_domain_prompt(domain),
                image,
                max_tokens=1600,
                deadline=deadline,
            )
        except ModelResponseError:
            domains[domain] = _failed_nemotron_domain(domain)
            failed_domains.append(domain)

    if progress_callback:
        progress_callback("Nemotron: generating final summary...")
    if time.monotonic() >= deadline:
        summary = _local_nemotron_summary(
            domains,
            "Final summary was generated locally because Nemotron reached the four-minute run limit.",
        )
    else:
        try:
            summary = _request_nemotron_json_step(
                "summary",
                build_nemotron_summary_prompt(domains),
                image,
                max_tokens=1000,
                deadline=deadline,
            )
        except ModelResponseError:
            summary = _local_nemotron_summary(
                domains,
                "Final summary was generated locally due to Nemotron failure.",
            )

    notes = str(summary.get("grading_notes", "")).strip()
    if failed_domains:
        failure_note = "Domains that failed to generate: " + ", ".join(failed_domains) + "."
        summary["grading_notes"] = f"{notes} {failure_note}".strip()
    return {**domains, **summary}


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned[:60] or "concept_map"


def run_evaluation(
    pdf_path: Path,
    model_names: Iterable[str],
    original_filename: str,
    progress_callback: Callable[[str], None] | None = None,
) -> list[EvaluationOutcome]:
    """Grade an uploaded PDF with each selected model and persist outcomes.

    Model-specific failures are returned as EvaluationFailure objects so a
    partial run can still show successful results from other models.
    """
    names = list(model_names)
    if not names:
        raise GradingError("Select at least one model.")
    unknown = [name for name in names if name not in MODEL_CONFIGS]
    if unknown:
        raise GradingError("Unknown model(s): " + ", ".join(unknown))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid4().hex[:8]
    file_stem = _safe_stem(original_filename)
    results: list[EvaluationOutcome] = []

    for model_name in names:
        model_id = MODEL_CONFIGS[model_name]["model_id"]
        image = render_pdf_image(pdf_path, model_name)
        raw_text: str | None = None
        try:
            if model_name == "Nemotron":
                returned_model_id = model_id
                data = _run_nemotron_sequential(image, progress_callback)
                data["map_file"] = Path(original_filename).name
                data["model"] = model_id
                data = normalize_nemotron_result(data)
                data = parse_model_json(json.dumps(data))
            else:
                prompt = build_model_prompt(
                    model_name, Path(original_filename).name, model_id
                )
                returned_model_id, raw_text = _request_model(
                    model_name, prompt, image
                )
                data = parse_model_json(raw_text)
            output_path = OUTPUT_DIR / (
                f"{timestamp}_{run_id}_{file_stem}_{model_name.lower()}.json"
            )
            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            results.append(
                EvaluationResult(model_name, returned_model_id, data, output_path)
            )
        except (ModelResponseError, MalformedResultError) as exc:
            raw_response = getattr(exc, "raw_response", None)
            if raw_response is None:
                raw_response = raw_text
            debug_path = _save_failed_response(
                timestamp=timestamp,
                run_id=run_id,
                file_stem=file_stem,
                model_name=model_name,
                model_id=model_id,
                error_message=str(exc),
                raw_response=raw_response,
            )
            results.append(
                EvaluationFailure(
                    model_name=model_name,
                    model_id=model_id,
                    error_message=str(exc),
                    debug_path=debug_path,
                )
            )

    return results
