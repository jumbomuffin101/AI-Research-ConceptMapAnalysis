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

MODEL = "google/gemma-4-26b-a4b-it:free"
PDF_PATH = "maps/ConceptMapWeak.pdf"
OUTPUT_PATH = "outputs/detected_content_weak_gemma.json"


def pdf_to_base64(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(2,2)
    )

    image_bytes = pix.tobytes("png")

    doc.close()

    return base64.b64encode(
        image_bytes
    ).decode("utf-8")


image = pdf_to_base64(PDF_PATH)

prompt = """
You are detecting the content of a student concept map.

Do not grade the map.

Your task is only to identify what is visible in the map.

Return valid JSON only.

Use this exact structure:

{
  "map_file": "ConceptMapStrong.pdf",
  "central_or_top_level_concepts": [],
  "detected_concepts": [],
  "detected_relationships": [
    {
      "source": "",
      "relationship_label": "",
      "target": ""
    }
  ],
  "hierarchy_detected": "",
  "layout_notes": "",
  "uncertain_items": []
}

Important rules:
1. A concept should usually be a short node label, not a full sentence.
2. A relationship should describe an arrow/link between two concepts.
3. If text is too long or unclear, put it in uncertain_items.
4. Do not invent missing links.
5. Do not evaluate quality.
6. Do not assign a score.
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":prompt
                },
                {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/png;base64,{image}"
                    }
                }
            ]
        }
    ]
)

result = response.choices[0].message.content

Path("outputs").mkdir(exist_ok=True)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(result)

print(result)
