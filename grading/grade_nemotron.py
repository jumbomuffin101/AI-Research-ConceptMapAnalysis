from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import fitz
from pathlib import Path
import re
import json
from datetime import datetime, timezone

load_dotenv()


def create_nvidia_client():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")
    return OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        timeout=300,
    )


def create_client():
    return create_nvidia_client()


MODEL = "microsoft/phi-4-multimodal-instruct"

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

MAPS = [
    {
        "label": "Map 1",
        "map_file": "ConceptMap1.pdf",
        "path": "maps/ConceptMap1.pdf",
        "output": "outputs/gradingV5/grounded_map1_phi4.json",
    },
    {
        "label": "Map 2",
        "map_file": "ConceptMap2.pdf",
        "path": "maps/ConceptMap2.pdf",
        "output": "outputs/gradingV5/grounded_map2_phi4.json",
    },
]


def pdf_to_base64(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB, alpha=False
        )
        image_bytes = pix.tobytes("jpeg", jpg_quality=80)
        return base64.b64encode(image_bytes).decode("utf-8")
    finally:
        doc.close()


def clean_json_output(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    return text


rubric = Path("rubric/concept_map_rubric.json").read_text(encoding="utf-8")


def _spring_rubric():
    rubric_data = json.loads(rubric)
    return {
        group: rubric_data[group]
        for group in CATEGORY_FIELDS
        if isinstance(rubric_data.get(group), dict)
    }


def _spring_schema(map_file):
    schema = {"map_file": map_file, "model": MODEL}
    for group, fields in CATEGORY_FIELDS.items():
        schema[group] = {
            field: {"score": 1, "explanation": "", "evidence_from_map": []}
            for field in fields
        }
        schema[group]["overall_decision"] = "No"
        schema[group]["if_no_explanation"] = ""
    schema.update(
        {
            "overall_meets_expectations": "No",
            "strengths": ["", ""],
            "areas_for_improvement": ["", ""],
            "grading_notes": "",
        }
    )
    return schema


def build_prompt(map_file):
    rubric_json = json.dumps(_spring_rubric(), separators=(",", ":"))
    schema_json = json.dumps(_spring_schema(map_file), separators=(",", ":"))
    return f"""Use the Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities exactly. Do not invent criteria.
Rubric:{rubric_json}
Rules: criterion scores are integers 1-4 only; domain overall_decision and overall_meets_expectations are exactly Yes or No; never use Partial, decimals, 0, or 5. Each criterion needs score, one short explanation, and brief evidence_from_map from visible content. Do not hallucinate. Each domain needs overall_decision and if_no_explanation when No. Keep strengths, areas_for_improvement, and grading_notes brief.
Return ONLY raw valid minified JSON matching this exact schema. No markdown or prose. First character {{; last character }}.
Schema:{schema_json}
"""


def request_grade(client, prompt, image=None):
    content = (
        [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            },
            {"type": "text", "text": prompt},
        ]
        if image is not None
        else prompt
    )
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=2500,
        messages=[{"role": "user", "content": content}],
    )


def _debug_response_text(response):
    if response is None:
        return ""
    method = getattr(response, "model_dump_json", None)
    if callable(method):
        return method(indent=2)
    return str(response)


def save_llama_raw_debug(path, response, raw_text, prompt, attempt):
    choices = getattr(response, "choices", None)
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    Path(path).write_text(
        json.dumps(
            {
                "provider": "NVIDIA NIM",
                "model": MODEL,
                "attempt": attempt,
                "raw_text": raw_text,
                "cleaned_text": clean_json_output(raw_text),
                "prompt_length": len(prompt),
                "max_tokens": 2500,
                "finish_reason": finish_reason,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _save_health_debug(
    path,
    payload_shape,
    response=None,
    error=None,
    prompt_length=0,
    image_file_size=0,
    extracted_terms=None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            {
                "provider": "NVIDIA NIM",
                "base_url": NVIDIA_BASE_URL,
                "model_id": MODEL,
                "prompt_length": prompt_length,
                "image_file_size": image_file_size,
                "payload_shape": payload_shape,
                "raw_response": _debug_response_text(response),
                "extracted_visible_terms": extracted_terms,
                "error": str(error) if error else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def extract_visible_terms(text):
    terms = []
    for candidate in re.split(r"[\r\n|;,]+", text):
        term = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", candidate).strip(" \t\"'")
        if not term or not re.search(r"[A-Za-z]", term):
            continue
        if term.lower().startswith(("here are", "visible words", "the image shows")):
            continue
        if term not in terms:
            terms.append(term)
    return terms[:10]


def extract_preflight_terms(text):
    parts = re.split(r"visible[_ ]terms\s*:", text, maxsplit=1, flags=re.IGNORECASE)
    term_text = parts[1] if len(parts) == 2 else text
    ignored = {
        "concept map",
        "medical concept map",
        "image",
        "nodes",
        "labels",
        "relationships",
    }
    terms = []
    for candidate in re.split(r"[\r\n|;,]+", term_text):
        term = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", candidate).strip(" \t\"'")
        if (
            not term
            or not re.search(r"[A-Za-z]", term)
            or len(term) > 100
            or term.lower() in ignored
        ):
            continue
        if term not in terms:
            terms.append(term)
    return terms[:10]


def run_health_tests(client, image, debug_prefix):
    """Run text and image checks before sending the full rubric prompt."""
    text_prompt = "Reply with OK."
    text_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text_prompt}],
        "temperature": 0,
        "max_tokens": 16,
    }
    text_path = f"{debug_prefix}_text_health.json"
    response = None
    try:
        response = client.chat.completions.create(**text_payload)
        content, reason = _response_content(response)
        if reason:
            raise RuntimeError(reason)
        _save_health_debug(
            text_path,
            text_payload,
            response=response,
            prompt_length=len(text_prompt),
        )
    except Exception as exc:
        _save_health_debug(
            text_path,
            text_payload,
            response=response,
            error=exc,
            prompt_length=len(text_prompt),
        )
        raise RuntimeError("NVIDIA text endpoint failed") from exc

    image_url_shape = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,<exact rendered image bytes>"
        },
    }
    preflight_prompt = (
        "Describe this concept map in one sentence. Then write VISIBLE_TERMS: "
        "followed by up to 10 medical concepts or terms that are clearly visible, "
        "separated by semicolons."
    )
    image_payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": preflight_prompt,
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 100,
    }
    image_payload_shape = {
        **image_payload,
        "messages": [
            {
                "role": "user",
                "content": [
                    image_url_shape,
                    {
                        "type": "text",
                        "text": preflight_prompt,
                    },
                ],
            }
        ],
    }
    image_path = f"{debug_prefix}_image_health.json"
    image_response = None
    image_file_size = len(base64.b64decode(image, validate=True))
    try:
        image_response = client.chat.completions.create(**image_payload)
        content, reason = _response_content(image_response)
        if reason:
            raise RuntimeError(reason)
        visible_terms = extract_preflight_terms(content)
        _save_health_debug(
            image_path,
            image_payload_shape,
            response=image_response,
            prompt_length=len(preflight_prompt),
            image_file_size=image_file_size,
            extracted_terms=visible_terms,
        )
        return content.strip(), visible_terms
    except Exception as exc:
        _save_health_debug(
            image_path,
            image_payload_shape,
            response=image_response,
            error=exc,
            prompt_length=len(preflight_prompt),
            image_file_size=image_file_size,
        )
        raise RuntimeError("NVIDIA image input failed") from exc


def _response_content(response):
    if response is None:
        return None, "response is None"
    choices = getattr(response, "choices", None)
    if not choices:
        return None, "response.choices is missing or empty"
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        return None, "choices[0].message is missing"
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        return None, "choices[0].message.content is empty"
    return content, ""


EVIDENCE_FIELDS = (
    "visible_diagnoses",
    "patient_data",
    "basic_science_concepts",
    "clinical_science_concepts",
    "health_system_science_concepts",
    "determinants_of_health",
    "visible_relationships_connections",
    "missing_or_unclear_required_elements",
)


def build_evidence_prompt():
    schema = {field: [] for field in EVIDENCE_FIELDS}
    return f"""Extract only evidence visibly present in this concept map image.
Do not grade the map. Do not infer from medical knowledge or add content that is not visible.
Extract visible diagnoses, patient data, basic science concepts, clinical science concepts,
health system science concepts, determinants of health, and visible relationships or connections.
Also identify required elements that are visibly missing or unclear within those categories.
Use short strings. If a category has no visible evidence, return an empty list.

Return ONLY valid JSON with this exact structure:
{json.dumps(schema, indent=2)}
"""


def extract_evidence(client, image, debug_prefix):
    prompt = build_evidence_prompt()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1200,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    Path(f"{debug_prefix}_evidence_raw_response.json").write_text(
        _debug_response_text(response), encoding="utf-8"
    )
    content, reason = _response_content(response)
    if reason:
        raise RuntimeError(f"Phi-4 evidence extraction failed: {reason}")
    evidence = json.loads(clean_json_output(content))
    missing = [field for field in EVIDENCE_FIELDS if field not in evidence]
    if missing:
        raise RuntimeError(
            "Phi-4 evidence extraction is missing fields: " + ", ".join(missing)
        )
    for field in EVIDENCE_FIELDS:
        if not isinstance(evidence[field], list) or not all(
            isinstance(item, str) for item in evidence[field]
        ):
            raise RuntimeError(f"Evidence field '{field}' must be a list of strings.")
    Path(f"{debug_prefix}_extracted_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    return evidence


COMPACT_LENGTHS = {"ka": 5, "int": 5, "app": 2, "tr": 3}
COMPACT_OVERALLS = (
    "ka_overall",
    "int_overall",
    "app_overall",
    "tr_overall",
    "overall",
)
SCORE_EXPLANATIONS = {
    1: "Little or irrelevant evidence was identified for this criterion.",
    2: "Evidence was partly relevant, too general, or limited.",
    3: "Evidence was relevant and mostly synthesized.",
    4: "Evidence was synthesized, detailed, and well-supported.",
}
DOMAIN_MAP = {
    "knowledge_acquisition": ("ka", "ka_overall", "ka", "Knowledge Acquisition"),
    "integration": ("int", "int_overall", "int", "Integration"),
    "application": ("app", "app_overall", "app", "Application"),
    "transfer": ("tr", "tr_overall", "tr", "Transfer"),
}


def build_compact_prompt(evidence):
    schema = {
        "ka": [1, 1, 1, 1, 1],
        "int": [1, 1, 1, 1, 1],
        "app": [1, 1],
        "tr": [1, 1, 1],
        "ka_overall": "No",
        "int_overall": "No",
        "app_overall": "No",
        "tr_overall": "No",
        "overall": "No",
        "evidence": {"ka": [], "int": [], "app": [], "tr": []},
        "improvements": [],
    }
    return f"""Independently grade the complete Spring 2025 concept map rubric using only the extracted visible evidence.
Do not infer from medical knowledge or grade what should be present.
Assign every criterion score as an integer 1-4 and every overall value as exactly "Yes" or "No".
The score arrays must follow the rubric criterion order shown below.
Keep at most one short evidence item per domain and at most two short improvements. Return compact minified JSON only. No explanations, markdown, or extra text.

Spring 2025 rubric:
{json.dumps(_spring_rubric(), separators=(",", ":"))}

Extracted evidence:
{json.dumps(evidence, separators=(",", ":"))}

Exact compact JSON structure:
{json.dumps(schema, separators=(",", ":"))}
"""


def validate_compact(data):
    for key, expected_length in COMPACT_LENGTHS.items():
        scores = data.get(key)
        if not isinstance(scores, list) or len(scores) != expected_length:
            raise RuntimeError(f"Compact field '{key}' has the wrong score count.")
        if any(
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 1 <= score <= 4
            for score in scores
        ):
            raise RuntimeError(f"Compact field '{key}' has invalid scores.")
    for key in COMPACT_OVERALLS:
        if data.get(key) not in {"Yes", "No"}:
            raise RuntimeError(f"Compact field '{key}' must be Yes or No.")


def compact_evidence(data, key):
    evidence = data.get("evidence")
    values = evidence.get(key) if isinstance(evidence, dict) else None
    if isinstance(values, list):
        cleaned = [item.strip() for item in values if isinstance(item, str) and item.strip()]
        if cleaned:
            return cleaned
    return ["No clear evidence found in the concept map."]


def expand_compact(data, map_file):
    result = _spring_schema(map_file)
    strengths = []
    for group, (score_key, overall_key, evidence_key, label) in DOMAIN_MAP.items():
        domain = result[group]
        domain_evidence = compact_evidence(data, evidence_key)
        for field, score in zip(CATEGORY_FIELDS[group], data[score_key]):
            domain[field] = {
                "score": score,
                "explanation": SCORE_EXPLANATIONS[score],
                "evidence_from_map": list(domain_evidence),
            }
        decision = data[overall_key]
        domain["overall_decision"] = decision
        domain["if_no_explanation"] = (
            f"{label} is marked No because Phi-4 identified insufficient visible evidence for this domain."
            if decision == "No"
            else ""
        )
        if decision == "Yes" and domain_evidence[0] != "No clear evidence found in the concept map.":
            strengths.append(domain_evidence[0])
    result["overall_meets_expectations"] = data["overall"]
    result["strengths"] = strengths[:2]
    result["areas_for_improvement"] = data.get("improvements", [])
    result["grading_notes"] = ""
    return result


PLAIN_SECTIONS = {
    "knowledge_acquisition": "Knowledge Acquisition",
    "integration": "Integration",
    "application": "Application",
    "transfer": "Transfer",
}
PLAIN_ALIASES = {
    "knowledge_acquisition": ("Knowledge Acquisition", "knowledge_acquisition", "KA"),
    "integration": ("Integration",),
    "application": ("Application",),
    "transfer": ("Transfer",),
    "final": ("Final", "Overall"),
}


def build_plain_prompt():
    lines = [
        "Independently grade every Spring 2025 rubric criterion using only the extracted visible evidence.",
        "Do not infer from medical knowledge or grade what should be present.",
        "Return plain text only, not JSON or markdown. Use exactly these headers and field names.",
        "Every score must be an integer 1-4. Every decision must be Yes or No. Keep each reason and list item short.",
        "",
    ]
    for group, header in PLAIN_SECTIONS.items():
        lines.append(f"{header}:")
        lines.extend(f"{field}: score 1-4" for field in CATEGORY_FIELDS[group])
        lines.extend(["overall_decision: Yes/No", "reason: short reason", ""])
    lines.extend(
        [
            "Final:",
            "overall_meets_expectations: Yes/No",
            "strengths: short item | short item",
            "areas_for_improvement: short item | short item | short item",
            "",
            "Spring 2025 rubric:",
            json.dumps(_spring_rubric(), separators=(",", ":")),
        ]
    )
    return "\n".join(lines)


def heading_pattern(aliases):
    variants = [re.escape(alias).replace(r"\ ", r"[\s_]+") for alias in aliases]
    return rf"(?im)^\s*(?:\#{{1,6}}\s*)?(?:\*\*)?(?:{'|'.join(variants)})\s*:?(?:\*\*)?\s*$"


def plain_section(text, section_key):
    headings = []
    for key, aliases in PLAIN_ALIASES.items():
        headings.extend(
            (match.start(), match.end(), key)
            for match in re.finditer(heading_pattern(aliases), text)
        )
    headings.sort()
    for index, (_, end, key) in enumerate(headings):
        if key == section_key:
            next_start = headings[index + 1][0] if index + 1 < len(headings) else len(text)
            return text[end:next_start]
    return None


def plain_score(section, field):
    field_pattern = re.escape(field).replace("_", r"[\s_-]+")
    match = re.search(
        rf"(?im)^\s*{field_pattern}\s*:\s*(?:score\s*)?(-?\d+(?:\.\d+)?)\s*$",
        section,
    )
    if not match:
        raise RuntimeError(f"Score '{field}' is missing.")
    value = match.group(1)
    if not re.fullmatch(r"[1-4]", value):
        raise RuntimeError(f"Score '{field}' must be an integer from 1 to 4.")
    return int(value)


def plain_decision(section, field, notes):
    match = re.search(rf"(?im)^\s*{re.escape(field)}\s*:\s*(Yes|No)\s*$", section)
    if match:
        return match.group(1)
    notes.append(f"Phi-4 omitted {field}; it defaulted to No.")
    return "No"


def specific_reason(reason, group=None):
    generic = {
        "meets expectations", "does not meet expectations", "insufficient evidence",
        "good", "complete", "all criteria met",
        "no clear evidence found in the concept map.",
    }
    normalized = reason.strip().lower()
    if len(reason.split()) < 6 or normalized in generic:
        return False
    domain_terms = {
        "knowledge_acquisition": ("science", "patient", "case", "data", "determinant", "diagnosis", "health"),
        "integration": ("connect", "link", "relationship", "differential", "illness", "patient data", "science"),
        "application": ("pathophysiology", "diagnosis", "patient data", "clinical", "science"),
        "transfer": ("prior", "learned", "transfer", "clinical", "basic science", "understanding"),
    }
    return group is None or any(term in normalized for term in domain_terms[group])


def parse_plain_grading(text, map_file):
    rubric_data = _spring_rubric()
    result = _spring_schema(map_file)
    notes = []
    global_decisions = re.findall(
        r"(?im)^\s*overall_decision\s*:\s*(Yes|No)\s*$", text
    )
    global_reasons = re.findall(r"(?im)^\s*reason\s*:\s*(.+?)\s*$", text)
    for index, group in enumerate(PLAIN_SECTIONS):
        section = plain_section(text, group)
        reason_match = (
            re.search(r"(?im)^\s*reason\s*:\s*(.+?)\s*$", section)
            if section is not None
            else None
        )
        reason = reason_match.group(1).strip() if reason_match else ""
        if not reason and index < len(global_reasons):
            reason = global_reasons[index].strip()
        evidence = [reason] if specific_reason(reason, group) else [
            "No clear evidence found in the concept map."
        ]
        for field in CATEGORY_FIELDS[group]:
            score = plain_score(text, field)
            descriptor = str(rubric_data[group][field][str(score)]).rstrip(".")
            result[group][field] = {
                "score": score,
                "explanation": f"Score {score}: {descriptor}.",
                "evidence_from_map": list(evidence),
            }
        decision_match = (
            re.search(r"(?im)^\s*overall_decision\s*:\s*(Yes|No)\s*$", section)
            if section is not None
            else None
        )
        if decision_match:
            decision = decision_match.group(1)
        elif index < len(global_decisions):
            decision = global_decisions[index]
        else:
            decision = "No"
            notes.append(f"Phi-4 omitted {group}.overall_decision; it defaulted to No.")
        result[group]["overall_decision"] = decision
        result[group]["if_no_explanation"] = (
            reason if decision == "No" and specific_reason(reason, group)
            else "The model did not provide a domain-level explanation." if decision == "No"
            else ""
        )
    final = plain_section(text, "final") or text
    result["overall_meets_expectations"] = plain_decision(
        final, "overall_meets_expectations", notes
    )
    for target, source, limit in (
        ("strengths", "strengths", 2),
        ("areas_for_improvement", "areas_for_improvement", 3),
    ):
        match = re.search(rf"(?im)^\s*{source}\s*:\s*(.*?)\s*$", final)
        result[target] = (
            [item.strip(" -\t") for item in re.split(r"\s*[|;]\s*", match.group(1)) if item.strip(" -\t")][:limit]
            if match and match.group(1).strip()
            else []
        )
    result["grading_notes"] = " ".join(notes)
    return result


def all_four_reasons_specific(result):
    for group, fields in CATEGORY_FIELDS.items():
        evidence = result[group][fields[0]].get("evidence_from_map", [])
        if not evidence or not specific_reason(str(evidence[0]), group):
            return False
    return True


def _is_all_four_result(result):
    scores = []
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            return False
        for field in fields:
            item = section.get(field)
            if not isinstance(item, dict):
                return False
            scores.append(item.get("score"))
    return bool(scores) and all(score == 4 for score in scores)


def _is_all_one_result(result):
    scores = []
    for group, fields in CATEGORY_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            return False
        for field in fields:
            item = section.get(field)
            if not isinstance(item, dict):
                return False
            scores.append(item.get("score"))
    return bool(scores) and all(score == 1 for score in scores)


def run_all():
    client = create_client()
    Path("outputs/gradingV5").mkdir(parents=True, exist_ok=True)
    debug_dir = Path("outputs/gradingV5/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for item in MAPS:
        image = pdf_to_base64(item["path"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        debug_prefix = debug_dir / f"{timestamp}_{Path(item['map_file']).stem}"
        image_debug_path = Path(f"{debug_prefix}_request.jpg")
        image_debug_path.write_bytes(base64.b64decode(image, validate=True))

        try:
            image_description, visible_terms = run_health_tests(
                client, image, debug_prefix
            )
            prompt = build_prompt(item["map_file"])
            prompt += (
                "\nImage preflight description: "
                + image_description
                + "\nExtracted visible medical terms: "
                + ("; ".join(visible_terms) if visible_terms else "None identified")
                + "\nCalibration rules: Use visible nodes, labels, and relationships as evidence. "
                "Do not require perfect OCR. Do not assign score 1 when relevant visible content exists. "
                "Score 1 only when the criterion is absent, irrelevant, or unreadable. "
                "Score 2 when content is visible but too general or weakly connected. "
                "Score 3 when content is relevant and mostly synthesized. "
                "Score 4 only when content is detailed, comprehensive, and clearly connected. "
                "Do not hallucinate evidence.\n"
            )
            Path(f"{debug_prefix}_grading_prompt.txt").write_text(
                prompt, encoding="utf-8"
            )
            response = request_grade(client, prompt, image)
            result, reason = _response_content(response)
            if reason:
                raise RuntimeError(f"NVIDIA grading request failed: {reason}")
            Path(f"{debug_prefix}_grading_raw_response.txt").write_text(
                result, encoding="utf-8"
            )
            save_llama_raw_debug(
                f"{debug_prefix}_phi4_attempt1_raw.json",
                response,
                result,
                prompt,
                1,
            )
            llama_attempt = 1
            try:
                parsed_result = json.loads(clean_json_output(result))
            except json.JSONDecodeError:
                llama_attempt += 1
                retry_prompt = (
                    f"{prompt}\n\nYour previous answer was not valid JSON. "
                    "Return the same evaluation as valid minified JSON only."
                )
                response = request_grade(client, retry_prompt, image)
                result, reason = _response_content(response)
                if reason:
                    raise RuntimeError(f"Phi-4 grading retry failed: {reason}")
                save_llama_raw_debug(
                    f"{debug_prefix}_phi4_attempt2_raw.json",
                    response,
                    result,
                    retry_prompt,
                    llama_attempt,
                )
                parsed_result = json.loads(clean_json_output(result))
            if len(visible_terms) >= 5 and _is_all_one_result(parsed_result):
                llama_attempt += 1
                calibration_prompt = (
                    f"{prompt}\n\nYou detected visible concept-map content. Regrade using "
                    "that visible evidence. Do not mark criteria as 1 unless truly absent."
                )
                response = request_grade(client, calibration_prompt, image)
                result, reason = _response_content(response)
                if reason:
                    raise RuntimeError(
                        f"NVIDIA grading request failed during calibration: {reason}"
                    )
                save_llama_raw_debug(
                    f"{debug_prefix}_phi4_calibration_raw.json",
                    response,
                    result,
                    calibration_prompt,
                    llama_attempt,
                )
                prompt = calibration_prompt
                parsed_result = json.loads(clean_json_output(result))
            cleaned_result = json.dumps(parsed_result, separators=(",", ":"))
            Path(f"{debug_prefix}_final_parsed_grading.json").write_text(
                json.dumps(parsed_result, indent=2), encoding="utf-8"
            )
            raw_output_path = item["output"].replace(".json", "_raw.txt")
            Path(raw_output_path).write_text(
                _debug_response_text(response),
                encoding="utf-8",
            )
            with open(item["output"], "w", encoding="utf-8") as f:
                json.dump(parsed_result, f, indent=2)

            print(f"\nSaved {item['output']}")
            print(cleaned_result)

        except Exception as e:
            message = str(e)
            if "NVCF asset pool must be given" in message:
                message = (
                    "NVIDIA NIM rejected the image payload format. The request "
                    "likely needs NVIDIA's asset upload/image format instead of "
                    "the current base64 image_url."
                )
            print(f"Error processing {item['label']}: {message}")
            continue


if __name__ == "__main__":
    run_all()
