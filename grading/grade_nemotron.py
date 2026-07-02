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


def request_grade(client, prompt):
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=3500,
        messages=[{"role": "user", "content": prompt}],
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


def build_evidence_grading_prompt(map_file, evidence):
    return f"""{build_prompt(map_file)}

Use ONLY the extracted evidence JSON below to grade. Do not use an image.
- Do not infer from medical knowledge.
- Do not grade based on what should be present.
- Score 4 only if the extracted evidence clearly supports the full rubric descriptor.
- Missing evidence must score 1.
- Vague or partial evidence must score 2.
- Relevant but incomplete evidence must score 3.
- Comprehensive and detailed evidence can score 4.

For every domain:
- overall_decision must be exactly "Yes" or "No".
- When overall_decision is "No", if_no_explanation must give a specific,
  domain-level reason grounded in the extracted evidence. Name the missing,
  unclear, or insufficient content or relationships; do not leave it empty or
  use a generic statement.
- When overall_decision is "Yes", if_no_explanation must be an empty string.
Example: "Integration is marked No because the concept map does not clearly connect patient data to clinical information or basic science."

Output brevity requirements:
- Each criterion explanation must be at most one short sentence.
- Each evidence_from_map must contain at most 1-2 short items.
- strengths must contain at most 2 short strings.
- areas_for_improvement must contain at most 3 short strings.
- grading_notes must be at most one short sentence.
- Return raw JSON only with no markdown or prose outside the JSON object.

Extracted evidence JSON:
{json.dumps(evidence, indent=2)}
"""


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
            evidence = extract_evidence(client, image, debug_prefix)
            grading_prompt = build_evidence_grading_prompt(
                item["map_file"], evidence
            )
            Path(f"{debug_prefix}_grading_prompt.txt").write_text(
                grading_prompt, encoding="utf-8"
            )
            outgoing_payload_shape = {
                "model": MODEL,
                "temperature": 0,
                "max_tokens": 3500,
                "messages": [
                    {
                        "role": "user",
                        "content": grading_prompt,
                    }
                ],
            }
            Path(f"{debug_prefix}_grading_payload.json").write_text(
                json.dumps(
                    {
                        "image_debug_path": str(image_debug_path),
                        "outgoing_payload_shape": outgoing_payload_shape,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            response = request_grade(client, grading_prompt)
            Path(f"{debug_prefix}_grading_raw_response.json").write_text(
                _debug_response_text(response), encoding="utf-8"
            )

            raw_output_path = item["output"].replace(".json", "_raw.txt")
            with open(raw_output_path, "w", encoding="utf-8") as f:
                f.write(str(response))

            result, reason = _response_content(response)
            if reason:
                print(f"Nemotron returned no usable content for {item['label']}: {reason}")
                continue

            cleaned_result = clean_json_output(result)
            try:
                parsed_result = json.loads(cleaned_result)
            except json.JSONDecodeError:
                Path(
                    f"{debug_prefix}_grading_malformed_raw_response.json"
                ).write_text(
                    _debug_response_text(response), encoding="utf-8"
                )
                retry_prompt = (
                    f"{grading_prompt}\n\n"
                    "Return minified JSON only. No markdown. No long "
                    "explanations. Keep all strings brief."
                )
                Path(f"{debug_prefix}_grading_retry_prompt.txt").write_text(
                    retry_prompt, encoding="utf-8"
                )
                retry_response = request_grade(client, retry_prompt)
                Path(
                    f"{debug_prefix}_grading_retry_raw_response.json"
                ).write_text(
                    _debug_response_text(retry_response), encoding="utf-8"
                )
                retry_result, retry_reason = _response_content(retry_response)
                if retry_reason:
                    raise RuntimeError(
                        f"Nemotron grading retry failed: {retry_reason}"
                    )
                cleaned_result = clean_json_output(retry_result)
                parsed_result = json.loads(cleaned_result)
            Path(f"{debug_prefix}_final_parsed_grading.json").write_text(
                json.dumps(parsed_result, indent=2), encoding="utf-8"
            )
            if _is_all_four_result(parsed_result):
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
