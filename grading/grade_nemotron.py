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
    )


MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

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


def _compact_spring_rubric_text():
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


def _compact_output_contract():
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


def build_prompt(map_file):
    return (
        "Spring 2025 SUMMATIVE concept map grading. Use only visible map evidence.\n"
        "Rules: score each criterion with integer 1-4 only; never use 0,5,decimals,"
        "Partial,Partially Meets,Borderline,Maybe. Domain overall_decision and "
        "overall_meets_expectations must be exactly Yes or No. Each criterion needs "
        "score, explanation, evidence_from_map. If evidence is missing, write "
        "\"No clear evidence found in the concept map.\" Do not hallucinate evidence. "
        "If a domain is No, fill if_no_explanation. Return only valid minified JSON.\n"
        f"Rubric:\n{_compact_spring_rubric_text()}\n"
        f"JSON contract: {_compact_output_contract()} "
        f'Use map_file="{map_file}" and model="{MODEL}".'
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
                        "text": prompt,
                    },
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
