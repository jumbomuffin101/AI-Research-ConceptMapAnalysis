from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import fitz
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "qwen/qwen2.5-vl-72b-instruct"

MAPS = [
    {
        "label": "strong",
        "map_file": "ConceptMapStrong.pdf",
        "path": "maps/ConceptMapStrong.pdf",
        "output": "outputs/graded_strong_qwen.json"
    },
    {
        "label": "weak",
        "map_file": "ConceptMapWeak.pdf",
        "path": "maps/ConceptMapWeak.pdf",
        "output": "outputs/graded_weak_qwen.json"
    }
]


def pdf_to_base64(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(image_bytes).decode("utf-8")


rubric = Path("rubric/concept_map_rubric.json").read_text(
    encoding="utf-8"
)

prompt_template = """
You are evaluating a student medical concept map using the rubric provided below.

Review the concept map and assign scores according to the rubric definitions.

Use evidence that is visible in the concept map when assigning scores.

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

Return valid JSON only using this exact structure:

{{
  "map_file": "{map_file}",
  "model": "{model}",
  "knowledge_acquisition": {{
    "basic_science": {{
      "score": 0,
      "reasoning": ""
    }},
    "health_system_science": {{
      "score": 0,
      "reasoning": ""
    }},
    "clinical_science": {{
      "score": 0,
      "reasoning": ""
    }},
    "patient_case_information": {{
      "score": 0,
      "reasoning": ""
    }},
    "determinants_of_health": {{
      "score": 0,
      "reasoning": ""
    }},
    "overall": {{
      "meets_expectations": "",
      "reasoning": ""
    }}
  }},
  "integration": {{
    "prioritized_differential_diagnosis": {{
      "score": 0,
      "reasoning": ""
    }},
    "illness_scripts": {{
      "score": 0,
      "reasoning": ""
    }},
    "basic_to_foundational_science": {{
      "score": 0,
      "reasoning": ""
    }},
    "patient_data_to_clinical_information": {{
      "score": 0,
      "reasoning": ""
    }},
    "patient_data_to_basic_science": {{
      "score": 0,
      "reasoning": ""
    }},
    "overall": {{
      "meets_expectations": "",
      "reasoning": ""
    }}
  }},
  "application": {{
    "working_diagnosis_pathophysiology": {{
      "score": 0,
      "reasoning": ""
    }},
    "patient_data_pathophysiology": {{
      "score": 0,
      "reasoning": ""
    }},
    "overall": {{
      "meets_expectations": "",
      "reasoning": ""
    }}
  }},
  "transfer": {{
    "prior_basic_science": {{
      "score": 0,
      "reasoning": ""
    }},
    "prior_clinical_concepts": {{
      "score": 0,
      "reasoning": ""
    }},
    "deepens_understanding": {{
      "score": 0,
      "reasoning": ""
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


Path("outputs").mkdir(exist_ok=True)

for item in MAPS:
    image = pdf_to_base64(item["path"])

    prompt = prompt_template.format(
        rubric=rubric,
        map_file=item["map_file"],
        model=MODEL
    )

    response = client.chat.completions.create(
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

    print(response)

    result = response.choices[0].message.content

    with open(item["output"], "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nSaved {item['output']}")
    print(result)