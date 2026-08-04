"""Direct Gemma grader for Spring 2025 concept map evaluation."""

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


class MalformedGemmaJsonError(RuntimeError):
    """Gemma grading was unusable after the permitted format-only recovery."""

    def __init__(self, message: str, attempts: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_response = attempts
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


def _format_repair_schema(map_file: str) -> dict[str, Any]:
    """Describe the production shape without numeric defaults that could anchor scores."""
    result: dict[str, Any] = {
        "map_file": map_file,
        "model": MODEL,
    }
    for group, fields in CATEGORY_FIELDS.items():
        result[group] = {
            **{
                field: {
                    "score": "<integer from 1 through 4>",
                    "explanation": "<preserve existing explanation>",
                }
                for field in fields
            },
            "overall_decision": "<Yes or No>",
            "if_no_explanation": "<preserve existing value>",
        }
    result["overall_meets_expectations"] = "<Yes or No>"
    result["strengths"] = ["<preserve each existing strength>"]
    result["areas_for_improvement"] = ["<preserve each existing improvement>"]
    result["grading_notes"] = "<preserve existing grading notes>"
    return result


def build_prompt(
    map_file: str, reference_materials: list[dict[str, str]] | None = None
) -> str:
    return build_grading_prompt(map_file, schema(map_file), reference_materials)


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


def request_format_repair(
    client: Any,
    raw_response: str,
    map_file: str,
) -> Any:
    """Request one text-only syntax repair without asking Gemma to regrade."""
    system_prompt = (
        "You are a deterministic JSON formatter. Convert the supplied completed grading "
        "response into valid JSON matching the required schema. Preserve every score, "
        "decision, explanation, strength, and improvement item. Return JSON only."
    )
    user_prompt = (
        "The previous Gemma grading response contains a complete evaluation but invalid "
        "JSON syntax.\n\nRepair the JSON formatting only.\n\nRules:\n"
        "- Do not re-evaluate the concept map.\n"
        "- Do not change any numeric score.\n"
        "- Do not add, remove, or reorder rubric criteria.\n"
        "- Do not change any domain decision.\n"
        "- Do not change overall_meets_expectations.\n"
        "- Do not invent missing evidence.\n"
        "- Do not summarize or shorten explanations unless required to escape invalid JSON characters.\n"
        "- Use valid double-quoted JSON.\n"
        "- Remove trailing commas.\n"
        "- Escape embedded quotes and control characters.\n"
        "- Return exactly one JSON object.\n"
        "- No Markdown fences.\n"
        "- No commentary before or after the JSON.\n\n"
        "RAW GEMMA RESPONSE:\n"
        + raw_response
        + "\n\nREQUIRED SCHEMA:\n"
        + json.dumps(_format_repair_schema(map_file), separators=(",", ":"))
    )
    return client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        timeout=TIMEOUT_SECONDS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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


def _response_metadata(response: Any) -> dict[str, Any]:
    dump = _response_debug_value(response)
    data = dump if isinstance(dump, dict) else {}
    choices = data.get("choices") if isinstance(data, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "finish_reason": first_choice.get("finish_reason")
        if isinstance(first_choice, dict)
        else None,
        "token_usage": usage,
        "response": dump,
    }


def _section_text(raw_text: str, group: str) -> str:
    groups = list(CATEGORY_FIELDS)
    start_match = re.search(rf'["\']{re.escape(group)}["\']\s*:', raw_text, re.IGNORECASE)
    if not start_match:
        return ""
    end = len(raw_text)
    group_index = groups.index(group)
    following_keys = groups[group_index + 1 :] + ["overall_meets_expectations"]
    for key in following_keys:
        match = re.search(
            rf'["\']{re.escape(key)}["\']\s*:',
            raw_text[start_match.end() :],
            re.IGNORECASE,
        )
        if match:
            end = min(end, start_match.end() + match.start())
    return raw_text[start_match.start() : end]


def _extract_identifiable_scores(raw_text: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for group, fields in CATEGORY_FIELDS.items():
        section = _section_text(raw_text, group)
        if not section:
            continue
        for index, field in enumerate(fields):
            field_match = re.search(
                rf'["\']{re.escape(field)}["\']\s*:', section, re.IGNORECASE
            )
            if not field_match:
                continue
            field_end = len(section)
            for next_field in fields[index + 1 :]:
                next_match = re.search(
                    rf'["\']{re.escape(next_field)}["\']\s*:',
                    section[field_match.end() :],
                    re.IGNORECASE,
                )
                if next_match:
                    field_end = field_match.end() + next_match.start()
                    break
            field_text = section[field_match.start() : field_end]
            score_match = re.search(
                r'["\']score["\']\s*:\s*["\']?([1-4])["\']?',
                field_text,
                re.IGNORECASE,
            )
            if score_match:
                scores[f"{group}.{field}"] = int(score_match.group(1))
    return scores


def _extract_identifiable_decisions(raw_text: str) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for group in CATEGORY_FIELDS:
        section = _section_text(raw_text, group)
        match = re.search(
            r'["\']overall_decision["\']\s*:\s*["\']([^"\'\r\n,}]+)',
            section,
            re.IGNORECASE,
        )
        if match:
            decisions[f"{group}.overall_decision"] = match.group(1).strip()
    overall = re.search(
        r'["\']overall_meets_expectations["\']\s*:\s*["\']([^"\'\r\n,}]+)',
        raw_text,
        re.IGNORECASE,
    )
    if overall:
        decisions["overall_meets_expectations"] = overall.group(1).strip()
    return decisions


def _parsed_score_snapshot(result: dict[str, Any]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            continue
        for field in fields:
            item = section.get(field)
            score = item.get("score") if isinstance(item, dict) else None
            if isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 4:
                scores[f"{group}.{field}"] = score
    return scores


def _parsed_decision_snapshot(result: dict[str, Any]) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for group in CATEGORY_FIELDS:
        section = result.get(group)
        if isinstance(section, dict) and isinstance(section.get("overall_decision"), str):
            decisions[f"{group}.overall_decision"] = section["overall_decision"].strip()
    if isinstance(result.get("overall_meets_expectations"), str):
        decisions["overall_meets_expectations"] = result["overall_meets_expectations"].strip()
    return decisions


def classify_grading_response(
    raw_text: str,
    response: Any,
) -> dict[str, Any]:
    """Classify whether malformed text contains a complete model-authored grading."""
    scores = _extract_identifiable_scores(raw_text)
    decisions = _extract_identifiable_decisions(raw_text)
    domains_present = [group for group in CATEGORY_FIELDS if _section_text(raw_text, group)]
    domains_missing = [group for group in CATEGORY_FIELDS if group not in domains_present]
    strengths_present = bool(
        re.search(
            r'["\']strengths["\']\s*:\s*\[.*?\]',
            raw_text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    improvements_present = bool(
        re.search(
            r'["\']areas_for_improvement["\']\s*:\s*\[.*?\]',
            raw_text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    metadata = _response_metadata(response)
    finish_reason = str(metadata.get("finish_reason") or "").lower()
    try:
        cleaned = clean_json_output(raw_text)
        json.loads(cleaned)
        parse_success = True
        parser_error = None
        parser_error_context = None
    except json.JSONDecodeError as exc:
        parse_success = False
        parser_error = str(exc)
        start = max(0, exc.pos - 240)
        end = min(len(cleaned), exc.pos + 240)
        parser_error_context = {
            "position": exc.pos,
            "line": exc.lineno,
            "column": exc.colno,
            "text": cleaned[start:end],
        }

    complete_content = (
        not domains_missing
        and len(scores) == sum(len(fields) for fields in CATEGORY_FIELDS.values())
        and len(decisions) == len(CATEGORY_FIELDS) + 1
        and strengths_present
        and improvements_present
    )
    if finish_reason == "length":
        classification = "truncated_grading_failure"
    elif parse_success and complete_content:
        classification = "valid_complete_grading"
    elif not parse_success and complete_content:
        classification = "format_only_failure"
    else:
        classification = "incomplete_grading_failure"
    return {
        "gemma_response_classification": classification,
        "initial_json_parse_success": parse_success,
        "initial_parser_error": parser_error,
        "initial_parser_error_context": parser_error_context,
        "initial_score_count": len(scores),
        "initial_all_scores_recoverable": len(scores) == 15,
        "initial_domains_present": domains_present,
        "initial_domains_missing": domains_missing,
        "initial_strengths_present": strengths_present,
        "initial_areas_for_improvement_present": improvements_present,
        "format_repair_eligible": classification == "format_only_failure",
        "initial_scores": scores,
        "initial_decisions": decisions,
        "finish_reason": metadata.get("finish_reason"),
        "token_usage": metadata.get("token_usage"),
        "raw_initial_response": raw_text,
        "initial_response_object": metadata.get("response"),
    }


def grade_pdf(
    pdf_path: Path,
    map_file: str,
    debug_prefix: Path,
    reference_materials: list[dict[str, str]] | None = None,
    progress_callback: Any | None = None,
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

    classification = classify_grading_response(raw_text, response)
    recovery_debug: dict[str, Any] = {
        **classification,
        "format_repair_attempted": False,
        "format_repair_image_resent": False,
        "repair_json_parse_success": False,
        "repair_schema_validation_success": False,
        "scores_preserved": None,
        "decisions_preserved": None,
        "raw_repair_response": None,
        "repair_response_object": None,
        "final_result_source": "initial"
        if classification["gemma_response_classification"]
        == "valid_complete_grading"
        else "failure",
    }

    if classification["gemma_response_classification"] == "format_only_failure":
        recovery_debug["format_repair_attempted"] = True
        if progress_callback:
            progress_callback(
                "Gemma completed the grading but returned malformed JSON. "
                "Repairing the response format…"
        )
        try:
            repair_response = request_format_repair(client, raw_text, map_file)
            recovery_debug["repair_response_object"] = _response_debug_value(
                repair_response
            )
            repair_text = response_text(repair_response)
            repair_raw_path = Path(f"{debug_prefix}_format_repair_raw.txt")
            repair_raw_path.write_text(repair_text, encoding="utf-8")
            recovery_debug["raw_repair_response"] = repair_text
            repaired = json.loads(clean_json_output(repair_text))
            recovery_debug["repair_json_parse_success"] = True
            if not isinstance(repaired, dict):
                raise ValueError("The Gemma format repair did not return a JSON object.")

            repaired_scores = _parsed_score_snapshot(repaired)
            repaired_decisions = _parsed_decision_snapshot(repaired)
            scores_preserved = repaired_scores == classification["initial_scores"]
            decisions_preserved = (
                repaired_decisions == classification["initial_decisions"]
            )
            recovery_debug["repair_scores"] = repaired_scores
            recovery_debug["repair_decisions"] = repaired_decisions
            recovery_debug["scores_preserved"] = scores_preserved
            recovery_debug["decisions_preserved"] = decisions_preserved
            if not scores_preserved:
                raise ValueError(
                    "Gemma format repair changed, added, or removed a rubric score."
                )
            if not decisions_preserved:
                raise ValueError("Gemma format repair changed a grading decision.")

            # Validate a copy through the existing production validator. Python does
            # not synthesize or replace any score during this check.
            from interface.grading_runner import parse_model_json

            parse_model_json(json.dumps(repaired), normalize_decisions=True)
            recovery_debug["repair_schema_validation_success"] = True
            recovery_debug["final_result_source"] = "format_repair"
            raw_text = repair_text
            response = repair_response
            raw_path = repair_raw_path
        except Exception as repair_error:
            recovery_debug["repair_error"] = str(repair_error)
            raise MalformedGemmaJsonError(
                "Gemma completed the grading, but its response could not be "
                "converted into valid grading JSON.",
                recovery_debug,
            ) from repair_error
    elif classification["gemma_response_classification"] == "truncated_grading_failure":
        raise MalformedGemmaJsonError(
            "Gemma's grading response was truncated before the complete JSON was returned.",
            recovery_debug,
        )
    elif classification["gemma_response_classification"] == "incomplete_grading_failure":
        raise MalformedGemmaJsonError(
            "The model response was not valid JSON: "
            + str(classification.get("initial_parser_error") or "incomplete grading response"),
            recovery_debug,
        )

    return {
        "model": MODEL,
        "provider": PROVIDER,
        "raw_text": raw_text,
        "cleaned_text": clean_json_output(raw_text),
        "response": response,
        "prompt": prompt,
        "prompt_path": prompt_path,
        "image_path": image_path,
        "image_base64": image_base64,
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
            **recovery_debug,
        },
    }
