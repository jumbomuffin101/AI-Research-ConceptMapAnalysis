from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import fitz
from pathlib import Path
import re
import json

load_dotenv()


def create_client():
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        timeout=300,
    )


MODEL = "google/gemma-4-26b-a4b-it:free"

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
        "output": "outputs/gradingV5/grounded_map1_gemma.json",
    },
    {
        "label": "Map 2",
        "map_file": "ConceptMap2.pdf",
        "path": "maps/ConceptMap2.pdf",
        "output": "outputs/gradingV5/grounded_map2_gemma.json",
    },
]


def pdf_to_base64(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(image_bytes).decode("utf-8")


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


def request_grade(client, prompt, image):
    return client.chat.completions.create(
        model=MODEL,
        # Reduced for OpenRouter free-tier compatibility on Streamlit Cloud
        max_tokens=2000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}"
                        },
                    },
                ],
            }
        ],
    )


def run_all():
    client = create_client()
    Path("outputs/gradingV5").mkdir(parents=True, exist_ok=True)

    for item in MAPS:
        image = pdf_to_base64(item["path"])
        prompt = build_prompt(item["map_file"])

        try:
            response = request_grade(client, prompt, image)

            raw_output_path = item["output"].replace(".json", "_raw.txt")
            with open(raw_output_path, "w", encoding="utf-8") as f:
                f.write(str(response))

            result = response.choices[0].message.content
            if result is None:
                print(f"No content returned for {item['label']}")
                continue

            cleaned_result = clean_json_output(result)
            with open(item["output"], "w", encoding="utf-8") as f:
                f.write(cleaned_result)

            print(f"\nSaved {item['output']}")
            print(cleaned_result)

        except Exception as e:
            print(f"Error processing {item['label']}: {e}")


if __name__ == "__main__":
    run_all()
