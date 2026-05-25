# AI Research Concept Map Analysis

This project investigates how multimodal AI models analyze medical concept maps and how model outputs differ in concept identification, relationship extraction, and hierarchy detection.

Current models:

- Qwen-VL (`qwen/qwen2.5-vl-72b-instruct`)
- Nemotron (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`)

Current tasks:

- Extract concepts from medical concept maps
- Detect relationships between concepts
- Identify hierarchical structure
- Compare model outputs
- Support future rubric-based grading experiments

Repository structure:

maps/
- Example strong and weak concept maps

outputs/
- Generated model outputs

Scripts:
- `detect_content_qwen.py`
- `detect_content_nemotron.py`