from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import fitz
from pathlib import Path
import re

load_dotenv()

def create_client():
    return OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

MAPS = [
    {
        "label": "Map 1",
        "map_file": "ConceptMap1.pdf",
        "path": "maps/ConceptMap1.pdf",
        "output": "outputs/gradingV4/grounded_map1_nemotron.json"
    },
    {
        "label": "Map 2",
        "map_file": "ConceptMap2.pdf",
        "path": "maps/ConceptMap2.pdf",
        "output": "outputs/gradingV4/grounded_map2_nemotron.json"
    }
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


rubric = Path("rubric/concept_map_rubric.json").read_text(
    encoding="utf-8"
)

prompt_template = """
You are evaluating a student medical concept map using the rubric provided below.

Review the concept map and assign scores according to the rubric definitions.

Use evidence that is visible in the concept map when assigning scores.

For each scored category, include 1-2 short evidence phrases copied from visible map text in evidence_from_map when possible.

If no visible evidence supports a category, write:
"No direct supporting evidence visible."

Provide brief reasoning for each score.

Rubric:
{rubric}

Case-specific expectations include:
- atrial fibrillation
- rapid ventricular response
- hypertension
- tobacco and alcohol use
- thyroid disease
- shock findings such as low blood pressure, clammy skin, poor pulses
- pulmonary edema
- need for immediate DC shock
- anticoagulation
- differential diagnosis
- illness scripts
- pathophysiology connecting atrial fibrillation, heart failure, and poor perfusion

Return ONLY raw valid JSON.
Do not include markdown.
Do not include ```json fences.
Do not include introductory text.
Do not include any text before or after the JSON object.

Return valid JSON only using this exact structure:

{{
  "map_file": "{map_file}",
  "model": "{model}",
  "knowledge_acquisition": {{
    "basic_science": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "health_system_science": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "clinical_science": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "patient_case_information": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "determinants_of_health": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "overall": {{
      "meets_expectations": "",
      "reasoning": ""
    }}
  }},
  "integration": {{
    "prioritized_differential_diagnosis": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "illness_scripts": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "basic_to_foundational_science": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "patient_data_to_clinical_information": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "patient_data_to_basic_science": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "overall": {{
      "meets_expectations": "",
      "reasoning": ""
    }}
  }},
  "application": {{
    "working_diagnosis_pathophysiology": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "patient_data_pathophysiology": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "overall": {{
      "meets_expectations": "",
      "reasoning": ""
    }}
  }},
  "transfer": {{
    "prior_basic_science": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "prior_clinical_concepts": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "deepens_understanding": {{
      "score": 0,
      "reasoning": "",
      "evidence_from_map": []
    }},
    "overall": {{
      "meets_expectations": "",
      "reasoning": ""
    }}
  }},
  "overall_map_meets_expectations": "",
  "strengths": [],
  "areas_for_improvement": [],
  "grading_notes": ""
}}

Important:
- Every numeric score must be an integer from 1 to 4.
- Use the rubric definitions as the source of truth for scoring.
- Do not add fields outside the requested JSON structure.

Scoring rules:
- Every numeric score must be an integer from 1 to 4.
- 1 = missing, incorrect, irrelevant, or minimal.
- 2 = partial, superficial, too general, or contains notable errors.
- 3 = relevant, mostly accurate, and mostly synthesized.
- 4 = detailed, comprehensive, accurate, and well-integrated.
"""


def build_prompt(map_file):
    return prompt_template.format(
        rubric=rubric,
        map_file=map_file,
        model=MODEL
    )


def request_grade(client, prompt, image):
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}"
                        }
                    }
                ]
            }
        ]
    )


def run_all():
    client = create_client()
    Path("outputs/gradingV4").mkdir(parents=True, exist_ok=True)

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
                print(response)
                continue

            cleaned_result = clean_json_output(result)

            with open(item["output"], "w", encoding="utf-8") as f:
                f.write(cleaned_result)

            print(f"\nSaved {item['output']}")
            print(cleaned_result)

        except Exception as e:
            print(f"Error processing {item['label']}: {e}")
            continue


if __name__ == "__main__":
    run_all()
