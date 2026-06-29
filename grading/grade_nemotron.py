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


MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

DOMAINS = [
    "knowledge_acquisition",
    "integration",
    "application",
    "transfer",
]

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


def _lightweight_schema(map_file):
    return {
        "map_file": map_file,
        "model": MODEL,
        "overall_meets_expectations": "No",
        "domain_scores": {domain: 1 for domain in DOMAINS},
        "brief_rationale": "",
        "agreement_notes": "",
    }


def build_prompt(map_file):
    return f"""Use the Spring 2025 concept map rubric at a high level.
Give an independent lightweight second opinion for the visible concept map.

Score only these four domains from 1 to 4:
{chr(10).join(f"- {domain}" for domain in DOMAINS)}

Rules:
- Scores must be integers 1, 2, 3, or 4 only.
- overall_meets_expectations must be exactly "Yes" or "No".
- Do not use Partial, Maybe, Borderline, score 0, score 5, or decimals.
- Keep brief_rationale and agreement_notes short.
- Return only valid JSON. No markdown. No prose outside JSON.

Do not grade every subcriterion.
Do not provide evidence_from_map.

Return exactly this JSON shape:
{json.dumps(_lightweight_schema(map_file), indent=2)}
"""


def request_grade(client, prompt, image):
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=800,
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

            result, reason = _response_content(response)
            if reason:
                print(f"Nemotron returned no usable content for {item['label']}: {reason}")
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
