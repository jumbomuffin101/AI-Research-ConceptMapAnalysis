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


MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

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
        "output": "outputs/gradingV5/grounded_map1_nemotron_vl.json",
    },
    {
        "label": "Map 2",
        "map_file": "ConceptMap2.pdf",
        "path": "maps/ConceptMap2.pdf",
        "output": "outputs/gradingV5/grounded_map2_nemotron_vl.json",
    },
]


def pdf_to_base64(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    try:
        for scale in (1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2):
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            image_bytes = pix.tobytes("png")
            if len(image_bytes) <= 180 * 1024:
                return base64.b64encode(image_bytes).decode("utf-8")
    finally:
        doc.close()
    raise RuntimeError(
        "The concept map image could not be reduced below NVIDIA's "
        "180 KB inline-image limit."
    )


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
    return f"""Use the Spring 2025 Concept Map Feedback Tool for SUMMATIVE Activities exactly.
Do not invent additional grading criteria.

Rubric:
{json.dumps(_spring_rubric(), indent=2)}

Global rules:
- Every criterion score must be an integer 1, 2, 3, or 4 only.
- Every domain overall_decision must be exactly "Yes" or "No".
- overall_meets_expectations must be exactly "Yes" or "No".
- Do not output Partial, Partially Meets, Borderline, Maybe, score 0, score 5, decimal scores, or any score outside 1-4.
- If evidence is missing, write "No clear evidence found in the concept map."
- Do not hallucinate evidence not visible in the concept map.

Each criterion must include score, explanation, and evidence_from_map.
Each domain must include overall_decision and if_no_explanation.
If overall_decision is "No", if_no_explanation is required.
The final overall decision answers: This map meets expectations.

Return ONLY raw valid JSON using this exact structure:
{json.dumps(_spring_schema(map_file), indent=2)}
"""


def request_grade(client, prompt, image=None):
    content = (
        [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"},
            },
            {"type": "text", "text": prompt},
        ]
        if image is not None
        else prompt
    )
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=2000,
        messages=[{"role": "user", "content": content}],
    )


def _debug_response_text(response):
    if response is None:
        return ""
    method = getattr(response, "model_dump_json", None)
    if callable(method):
        return method(indent=2)
    return str(response)


def _save_health_debug(path, payload_shape, response=None, error=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            {
                "provider": "NVIDIA NIM",
                "model_id": MODEL,
                "payload_shape": payload_shape,
                "raw_response": _debug_response_text(response),
                "error": str(error) if error else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_health_tests(client, image, debug_prefix):
    """Run text and image checks before sending the full rubric prompt."""
    text_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0,
        "max_tokens": 16,
    }
    text_path = f"{debug_prefix}_text_health.json"
    try:
        response = client.chat.completions.create(**text_payload)
        content, reason = _response_content(response)
        if reason:
            raise RuntimeError(f"Nemotron text health test failed: {reason}")
        _save_health_debug(text_path, text_payload, response=response)
    except Exception as exc:
        _save_health_debug(text_path, text_payload, error=exc)
        raise

    image_url_shape = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,<exact rendered image bytes>"
        },
    }
    image_payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in one sentence.",
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
                        "text": "Describe this image in one sentence.",
                    },
                ],
            }
        ],
    }
    image_path = f"{debug_prefix}_image_health.json"
    try:
        response = client.chat.completions.create(**image_payload)
        content, reason = _response_content(response)
        if reason:
            raise RuntimeError(f"Nemotron image health test failed: {reason}")
        _save_health_debug(image_path, image_payload_shape, response=response)
    except Exception as exc:
        _save_health_debug(image_path, image_payload_shape, error=exc)
        raise


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
                            "url": f"data:image/png;base64,{image}"
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
        raise RuntimeError(f"Nemotron evidence extraction failed: {reason}")
    evidence = json.loads(clean_json_output(content))
    missing = [field for field in EVIDENCE_FIELDS if field not in evidence]
    if missing:
        raise RuntimeError(
            "Nemotron evidence extraction is missing fields: " + ", ".join(missing)
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
            f"{label} is marked No because Nemotron identified insufficient visible evidence for this domain."
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
    notes.append(f"Nemotron omitted {field}; it defaulted to No.")
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
            notes.append(f"Nemotron omitted {group}.overall_decision; it defaulted to No.")
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


def run_all():
    client = create_client()
    Path("outputs/gradingV5").mkdir(parents=True, exist_ok=True)
    debug_dir = Path("outputs/gradingV5/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    for item in MAPS:
        image = pdf_to_base64(item["path"])
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        debug_prefix = debug_dir / f"{timestamp}_{Path(item['map_file']).stem}"
        image_debug_path = Path(f"{debug_prefix}_request.png")
        image_debug_path.write_bytes(base64.b64decode(image, validate=True))

        try:
            run_health_tests(client, image, debug_prefix)
            plain_prompt = build_plain_prompt()
            Path(f"{debug_prefix}_plain_grading_prompt.txt").write_text(
                plain_prompt, encoding="utf-8"
            )
            response = request_grade(client, plain_prompt, image)
            result, reason = _response_content(response)
            if reason:
                raise RuntimeError(f"Nemotron plain-text grading failed: {reason}")
            Path(f"{debug_prefix}_plain_grading_raw_response.txt").write_text(
                result, encoding="utf-8"
            )
            try:
                parsed_result = parse_plain_grading(result, item["map_file"])
            except RuntimeError:
                retry_prompt = (
                    "Use the exact headings and field names shown below. Do not rename, omit, or reformat them.\n\n"
                    f"{plain_prompt}"
                )
                response = request_grade(client, retry_prompt, image)
                retry_result, retry_reason = _response_content(response)
                if retry_reason:
                    raise RuntimeError(
                        f"Nemotron plain-text grading retry failed: {retry_reason}"
                    )
                Path(
                    f"{debug_prefix}_plain_grading_retry_raw_response.txt"
                ).write_text(retry_result, encoding="utf-8")
                parsed_result = parse_plain_grading(
                    retry_result, item["map_file"]
                )
            cleaned_result = json.dumps(parsed_result, separators=(",", ":"))
            Path(f"{debug_prefix}_final_parsed_grading.json").write_text(
                json.dumps(parsed_result, indent=2), encoding="utf-8"
            )
            raw_output_path = item["output"].replace(".json", "_raw.txt")
            Path(raw_output_path).write_text(
                _debug_response_text(response),
                encoding="utf-8",
            )
            if (
                _is_all_four_result(parsed_result)
                and not all_four_reasons_specific(parsed_result)
            ):
                raise RuntimeError(
                    "Nemotron returned an implausible all-4 evaluation. "
                    "Raw output saved for debugging."
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
