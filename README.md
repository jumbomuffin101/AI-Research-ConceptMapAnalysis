# AI Concept Map Evaluation System

AI-powered concept map evaluation system for medical education.

The project evaluates medical concept maps using multimodal AI models and presents rubric-aligned grading results in an interactive Streamlit dashboard.

## Status

**Status: Working Prototype**

Completed:

- [x] Dual-model grading
- [x] Structured JSON output
- [x] Web interface
- [x] PDF processing pipeline
- [x] Rubric-based evaluation

In Progress:

- [ ] Expanded validation dataset
- [ ] Inter-model agreement analytics

## Current Pipeline

```text
Concept Map PDF
    |
    v
PDF-to-Image Conversion
    |
    v
Vision Language Model
    (Gemma via OpenRouter OR Qwen 3.6 27B via Groq)
    |
    v
Rubric-Based Grading
    |
    v
Structured JSON Output
    |
    v
Interactive Web Dashboard
```

## Completed Features

- PDF concept map upload
- PDF-to-image conversion
- Dual-model evaluation (Gemma + Qwen 3.6 27B)
- Direct rubric-based grading
- Rubric-based scoring
- Knowledge Acquisition grading
- Integration grading
- Application grading
- Transfer grading
- Strength identification
- Areas for improvement generation
- Structured JSON export
- Interactive web dashboard
- Multi-model comparison mode
- Graceful handling of single-model failures

## Model Information

### Primary Models

- `google/gemma-4-26b-a4b-it:free`
- `qwen/qwen3.6-27b`

Gemma uses OpenRouter. Qwen 3.6 27B uses Groq. Both models generate full Spring 2025 rubric-aligned JSON grading.

Required environment variables:

- `OPENROUTER_API_KEY` for Gemma
- `GROQ_API_KEY` for Qwen 3.6 27B

Users may run:

- Gemma only
- Qwen 3.6 27B only
- Both models simultaneously

## Repository Structure

```text
maps/
|-- ConceptMapStrong.pdf
`-- ConceptMapWeak.pdf

grading/
|-- grade_gemma.py
`-- grade_llama.py

interface/
|-- __init__.py
|-- grading_runner.py
`-- result_display.py

rubric/
`-- concept_map_rubric.json

outputs/
|-- gradingV1/
|-- gradingV2/
|-- gradingV3/
|-- gradingV4/
|-- gradingV5/
`-- web_demo/

app.py
requirements.txt
runtime.txt
```

## Web Demo

The Streamlit demo accepts any concept map PDF, runs Gemma, Qwen 3.6 27B, or both, and displays rubric scores, reasoning, evidence, strengths, and areas for improvement. Valid results are saved under `outputs/web_demo/`.

If one selected model fails, the app keeps any successful model result visible and shows a warning for the failed model. Raw failed responses are saved under `outputs/web_demo/debug/` for troubleshooting.

Install dependencies and configure API keys:

```powershell
pip install -r requirements.txt
$env:OPENROUTER_API_KEY="your-api-key"
$env:GROQ_API_KEY="your-groq-api-key"
```

You can instead place these values in the project `.env` file or Streamlit secrets:

```text
OPENROUTER_API_KEY=your-openrouter-api-key
GROQ_API_KEY=your-groq-api-key
```

Start the app from the repository root:

```powershell
streamlit run app.py
```

## Command-Line Usage

Run the grading scripts from the repository root:

```powershell
python grading/grade_gemma.py
python grading/grade_llama.py
```

New grading outputs are organized under `outputs/gradingV5/`. Historical outputs in earlier grading folders are preserved for comparison.
