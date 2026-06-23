# AI Research Concept Map Analysis

This project investigates how AI-generated concept map evaluations differ from human-generated grading and how collaborative grading strategies compare across multiple AI and human evaluators.

The current system uses multimodal vision-language models to analyze medical concept maps through concept extraction, relationship detection, hierarchy identification, and rubric-based grading. The long-term goal is to compare AI-generated evaluations against human-generated evaluations and explore multi-AI and AI-human collaborative grading workflows.

## Research Objectives

* Compare AI-generated evaluations with human-generated grading
* Investigate multi-AI grading workflows
* Investigate AI-human collaborative grading workflows
* Evaluate agreement, disagreement, and consensus strategies
* Support automated rubric-based assessment of medical concept maps

## Current Models

### Content Extraction and Grading

* Qwen-VL (`qwen/qwen2.5-vl-72b-instruct`)
* Nemotron (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`)

## Current Progress

### Completed

* Concept extraction from medical concept maps
* Relationship detection between concepts
* Hierarchy identification
* Structured JSON output generation
* Comparison of model extraction outputs
* Custom rubric implementation
* Rubric-based grading of Strong and Weak concept maps
* Prompting methodology evaluation
* Comparison of grading outputs across models

### In Progress

* Multi-AI consensus grading
* AI-human collaborative grading
* Comparison against human-generated scores
* Expanded testing on additional concept maps

## Repository Structure

```text
maps/
├── ConceptMapStrong.pdf
└── ConceptMapWeak.pdf

extraction/
├── detect_content_qwen.py
└── detect_content_nemotron.py

grading/
├── grade_qwen.py
└── grade_nemotron.py

rubric/
└── concept_map_rubric.json

outputs/
├── gradingV1/
├── gradingV2/
├── gradingV3/
└── extraction outputs
```

## Current Pipeline

```text
Concept Map PDF
        ↓
PDF-to-Image Conversion
        ↓
Vision-Language Model
(Qwen-VL / Nemotron)
        ↓
Concept Extraction
        ↓
Relationship Detection
        ↓
Rubric-Based Grading
        ↓
Structured JSON Output
```

## Current Findings

* Both Qwen-VL and Nemotron successfully identify concept map structure and major concepts.
* Qwen generally produces more detailed extraction outputs.
* Nemotron generally produces more concise outputs.
* Both models successfully distinguish Strong and Weak concept maps using the grading rubric.
* Model agreement was higher during grading than during initial content extraction.

## Web Demo

The Streamlit demo accepts any concept map PDF, runs Qwen, Nemotron, or both through
OpenRouter, and displays rubric scores, reasoning, evidence, strengths, and areas for
improvement. Valid results are saved under `outputs/web_demo/`.

If one selected model fails, the app keeps any successful model result visible and
shows a warning for the failed model. Raw failed responses are saved under
`outputs/web_demo/debug/` for troubleshooting. Nemotron may intermittently fail to
return usable JSON on dense concept maps; when that happens, select `Nemotron` only
and run the evaluation again to retry it independently.

Install the dependencies and configure the OpenRouter API key:

```powershell
pip install -r requirements.txt
$env:OPENROUTER_API_KEY="your-api-key"
```

You can instead place `OPENROUTER_API_KEY=your-api-key` in the project `.env` file.
Then start the app from the repository root:

```powershell
streamlit run app.py
```
