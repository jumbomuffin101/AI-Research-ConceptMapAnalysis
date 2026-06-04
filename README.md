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
